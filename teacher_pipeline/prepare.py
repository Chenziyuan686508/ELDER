"""Full selected-file audit, media grouping, shared A/B manifests and batch plan."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import random
from .common import parser, load_config, iter_json_array, write_json, write_jsonl, digest, file_digest
from .data import MediaResolver, normalize, UnionFind


def prepare(c):
    out = Path(c['output']); d = c['data']; rng = random.Random(c['experiment']['seed'])
    if any((out/'runs').glob('*/latest.json')):
        raise RuntimeError('Cannot replace manifests after pilot training starts; use a new experiment output directory')
    resolver = MediaResolver(d['media_roots']); uf = UnionFind()
    rows, rejected, report = [], [], {}
    for task in d['selected_tasks']:
        path = Path(d['root']) / (task + '.json')
        counts = Counter(); seen = set(); samples = []; flags = Counter()
        for i, raw in enumerate(iter_json_array(path)):
            counts['scanned'] += 1
            try:
                row = normalize(raw, task, i, resolver)
                if row['sample_id'] in {'MSCOCO_i2t.json:19145','Visual7W.json:1489'}:
                    raise ValueError('reviewed_cot_instruction_contradiction')
                pair = (row['query']['id'], row['candidate']['id'])
                if pair in seen: raise ValueError('duplicate_pair')
                seen.add(pair)
                for k in row['group_keys']: uf.union(row['group_keys'][0], k)
                rows.append(row); counts['valid_before_candidate_audit'] += 1
                flags.update(row['quality_flags'])
                # Reservoir, never file-prefix sampling.
                if len(samples) < 20: samples.append(row)
                else:
                    j = rng.randrange(counts['valid_before_candidate_audit'])
                    if j < 20: samples[j] = row
            except ValueError as e:
                reason = str(e); counts['removed_' + reason] += 1
                rejected.append(dict(sample_id=f'{task}.json:{i}', reason=reason))
            if (i+1) % 5000 == 0: print(task, i+1, dict(counts), flush=True)
        report[task] = dict(counts=counts, quality_flags=flags, source_sha256=file_digest(path))
        write_jsonl(out/'review'/f'{task}.jsonl', samples)
    # Choose candidate CoT independently from all occurrences; conservative lexical risk gate.
    by_candidate = defaultdict(list)
    for r in rows: by_candidate[(r['task'],r['candidate']['id'])].append(r)
    candidates = {}; removed = set(); merge_audit = []
    for key, occurrences in by_candidate.items():
        choices = [r for r in occurrences if 'candidate_possible_pair_conditioning' not in r['quality_flags']]
        if not choices:
            removed.add(key)
        else:
            chosen = min(choices, key=lambda r: (digest(r['candidate']['cot']),r['sample_id']))
            candidates[key] = chosen['candidate']
        if len(occurrences)>1 or not choices:
            merge_audit.append(dict(task=key[0], candidate_id=key[1], occurrences=len(occurrences),
                                    distinct_cots=len({r['candidate']['cot'] for r in occurrences}),
                                    selected_source=min(choices,key=lambda r:(digest(r['candidate']['cot']),r['sample_id']))['sample_id'] if choices else None,
                                    rule='lexical risk gate, then minimum CoT SHA256; provenance remains unknown'))
    clean=[]
    for r in rows:
        key=(r['task'],r['candidate']['id'])
        if key in removed:
            rejected.append(dict(sample_id=r['sample_id'],reason='candidate_no_low_risk_cot'));continue
        r['candidate']=candidates[key]
        r['source_group_id']=digest(uf.find(r['group_keys'][0]))
        r.pop('group_keys')
        r['split']='validation' if int(r['source_group_id'][:12],16)/16**12 < d['validation_group_fraction'] else 'train'
        clean.append(r)
    # Known positives shared by exact query, or same source image in caption retrieval.
    positives=defaultdict(set)
    def relevance_key(r):
        return (r['task'],r['source_group_id'] if r['task'].startswith('MSCOCO') else r['query']['id'])
    for r in clean: positives[relevance_key(r)].add(r['candidate']['id'])
    for r in clean: r['positive_candidate_ids']=sorted(positives[relevance_key(r)])
    train=[]; valid=[]; plan=[]
    cap=d['max_train_pairs']//len(d['selected_tasks'])
    for task in d['selected_tasks']:
        tr=[r for r in clean if r['task']==task and r['split']=='train'];rng.shuffle(tr);tr=tr[:cap]
        va=[r for r in clean if r['task']==task and r['split']=='validation']
        # Deduplicate validation query; all relevant candidates retained in full held-out corpus.
        unique={}
        for r in va: unique.setdefault(r['query']['id'],r)
        queries=list(unique.values());rng.shuffle(queries);queries=queries[:d['validation_queries_per_task']]
        corpus={r['candidate']['id']:r['candidate'] for r in va}
        write_jsonl(out/'data'/f'{task}.candidates.jsonl',corpus.values())
        train.extend(tr);valid.extend(queries)
        report[task]['split']=dict(train_available=sum(r['task']==task and r['split']=='train' for r in clean),
                                  train_selected=len(tr),validation_queries=len(queries),validation_candidates=len(corpus))
    rng.shuffle(train)
    batch_size=c['training']['contrastive_pool_target_pairs']
    for task in d['selected_tasks']:
        ids=[r['sample_id'] for r in train if r['task']==task]
        for i in range(0,len(ids),batch_size):
            b=ids[i:i+batch_size]
            if len(b)>1: plan.append(dict(task=task,sample_ids=b))
    rng.shuffle(plan);plan=plan[:c['training']['max_optimizer_steps']]
    tg={r['source_group_id'] for r in train};vg={r['source_group_id'] for r in valid}
    assert not tg & vg
    report['summary']=dict(train_pairs=len(train),validation_queries=len(valid),optimizer_steps=len(plan),
        unique_media_verified=len(resolver.cache),group_overlap=0,selected_files_full_scan=True,
        all_39_files_scanned=False,provenance='unknown; lexical checks do not prove independence',
        source_revision_verified=False,source_revision_note='requested revision recorded; local bytes fingerprinted, remote equality not yet checked',
        manual_review='random 20/task exported; review before training',
        excluded_modalities=['video','visual_document'],candidate_cot_rejected=len(removed))
    write_jsonl(out/'data'/'train.jsonl',train);write_jsonl(out/'data'/'validation.jsonl',valid)
    write_jsonl(out/'data'/'filtered.jsonl',rejected);write_jsonl(out/'data'/'candidate_merges.jsonl',merge_audit)
    reference=out/'reference'/'data_revision_files.json'
    if reference.exists():
        remote=json.loads(reference.read_text())
        checked={Path(x['path']).stem:x['lfs']['oid'] for x in remote}
        report['summary']['source_revision_verified']=all(checked.get(task)==report[task]['source_sha256'] for task in d['selected_tasks'])
        if report['summary']['source_revision_verified']:
            report['summary']['source_revision_note']='Selected files match remote LFS SHA256 at requested immutable revision'
    for task in d['selected_tasks']:
        report[task]['all_removal_counts']=dict(Counter(r['reason'] for r in rejected if r['sample_id'].startswith(task+'.json:')))
    report['summary']['heldout_scope']='Held out from pilot fine-tuning; released retrieval base may have seen the original training sources'
    write_json(out/'data'/'batch_plan.json',plan);write_json(out/'data_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    args=parser(__doc__).parse_args();prepare(load_config(args.config))

if __name__=='__main__': main()
