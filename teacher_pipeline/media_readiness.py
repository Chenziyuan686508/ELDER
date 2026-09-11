"""Audit video/document media and exercise real teacher inputs; never filter CoT quality.

Outputs are isolated from the pilot train/validation manifests and checkpoints.
Document mapping uses the encoded global source-row index AND exact source query
(and answer for ColPali), not filename suffix guesses or query-only fuzzy matching.
Extracted PNGs preserve original decoded RGB pixels, not the author's JPEG bytes.
"""
import argparse, collections, copy, io, json, random, re
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from PIL import Image
from .common import iter_json_array, write_json, write_jsonl, load_config, digest
from .data import MediaResolver, normalize

ROOT=Path('/root/autodl-tmp/datasets')
SOURCE=ROOT/'Embed-RL-Train/MMEB-train-contrastive-learning'
OUT=Path('/root/autodl-tmp/eval/elder/media_readiness')
PREFIX='Align the question with <|image_1|> for representation: '

def source_index(kind):
    entries=[]
    if kind=='vidore':
        for path in sorted((ROOT/'colpali_train_set').glob('data-*-of-*.arrow')):
            with pa.memory_map(str(path)) as f:
                for bi,batch in enumerate(ipc.open_stream(f)):
                    for ri,row in enumerate(batch.select(['query','answer']).to_pylist()):
                        entries.append((row, str(path), bi,ri))
    else:
        for path in sorted((ROOT/'VisRAG-Ret-Train-In-domain-data/data').glob('train-?????-of-?????.parquet')):
            for ri,row in enumerate(pq.read_table(path,columns=['query']).to_pylist()):
                entries.append((row,str(path),None,ri))
    return entries

def extract(entry, path):
    row,src,bi,ri=entry
    if bi is None:
        f=pq.ParquetFile(src); offset=ri
        for gi in range(f.num_row_groups):
            n=f.metadata.row_group(gi).num_rows
            if offset<n:
                raw=f.read_row_group(gi,columns=['image']).slice(offset,1).to_pylist()[0]['image'];break
            offset-=n
    else:
        with pa.memory_map(src) as f:
            for i,b in enumerate(ipc.open_stream(f)):
                if i==bi:
                    raw=b.select(['image']).slice(ri,1).to_pylist()[0]['image'];break
    with Image.open(io.BytesIO(raw['bytes'])) as im:
        im=im.convert('RGB'); im.save(path,format='PNG')
        return dict(source=src,batch=bi,row=ri,size=list(im.size),rgb_sha256=digest([im.size,__import__('hashlib').sha256(im.tobytes()).hexdigest()]))

def audit():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'media').mkdir(exist_ok=True)
    rng=random.Random(42);samples={};seen=collections.Counter();report={};exists={}
    def offer(task,value):
        seen[task]+=1;bucket=samples.setdefault(task,[])
        if len(bucket)<4:bucket.append(value)
        else:
            j=rng.randrange(seen[task])
            if j<4:bucket[j]=value
    for path in sorted(SOURCE.glob('llavahound*.json')):
        counts=collections.Counter()
        for i,r in enumerate(iter_json_array(path)):
            counts['records']+=1;frames=[]
            for side in ['qry','pos']:
                frames += [p for p in r[side].get('video',[]) if p and p!='images/blank.jpg']
            partitions={p.split('/')[3] for p in frames}
            for p in partitions:counts[p]+=1
            if not frames:counts['no_frames']+=1;continue
            ok=True
            for p in frames:
                local=ROOT/'train_video_and_instruction'/p.split('train_video_and_instruction/',1)[1]
                if str(local) not in exists:exists[str(local)]=local.is_file()
                ok=exists[str(local)] and ok
            counts['all_frames_exist' if ok else 'missing_frames']+=1
            if ok:offer(r['dataset_name'],(path.name,i,r,None))
        report[path.name]=dict(counts);print(path.name,dict(counts),flush=True)
        write_json(OUT/'scan_progress.json',report)
    for kind in ['vidore','visrag']:
        index=source_index(kind);print(kind,'source_rows',len(index),flush=True)
        for path in sorted(SOURCE.glob(kind+'_samples_*.json')):
            counts=collections.Counter();examples=[]
            for i,r in enumerate(iter_json_array(path)):
                counts['records']+=1
                media=r['qry_image_path'] if kind=='vidore' else r['pos_image_path']
                m=re.fullmatch(r'images/[^/]+/'+kind+r'_(\d+)_[0-9a-f]+\.(jpeg|png|jpg)',media)
                if not m:counts['unrecognized_filename']+=1;continue
                idx=int(m[1])
                if idx>=len(index):counts['index_out_of_range']+=1;continue
                original=index[idx][0]
                query=r['qry']
                matches=query.startswith(PREFIX) and query[len(PREFIX):].strip()==original['query'].strip()
                if kind=='vidore':matches=matches and r['pos_text'].strip()==original['answer'].strip()
                if not matches:
                    counts['source_content_mismatch']+=1
                    if len(examples)<3:examples.append(dict(index=idx,annotation=query,source=original))
                    continue
                counts['source_index_and_content_match']+=1
                offer(kind,(path.name,i,r,index[idx]))
            report[path.name]=dict(counts,source_rows=len(index),mismatch_examples=examples)
            print(path.name,report[path.name],flush=True)
        write_json(OUT/'scan_progress.json',report)
    c=load_config('configs/teacher_pilot.yaml');resolver=MediaResolver(c['data']['media_roots'])
    normalized=[];provenance=[]
    for task,bucket in samples.items():
        for file,i,raw,entry in bucket:
            raw=copy.deepcopy(raw)
            if entry:
                media=OUT/'media'/f'{task}_{i}_{digest(file)[:8]}.png'
                proof=extract(entry,media)
                resolver.roots.insert(0,('readiness/',str(OUT/'media')))
                raw['qry_image_path' if task=='vidore' else 'pos_image_path']='readiness/'+media.name
                provenance.append(dict(annotation=file,annotation_row=i,media=str(media),**proof))
            try:
                row=normalize(raw,task,i,resolver);row['source_file']=file;row['sample_id']=f'{file}:{i}'
                normalized.append(row)
            except Exception as e:
                report.setdefault('sample_read_errors',[]).append(dict(file=file,row=i,error=repr(e)))
    write_jsonl(OUT/'samples.jsonl',normalized)
    write_json(OUT/'document_provenance.json',provenance)
    report['summary']=dict(sample_counts=dict(collections.Counter(r['task'] for r in normalized)),
        unique_video_paths_checked=len(exists),unique_video_paths_present=sum(exists.values()),
        sample_media_decoded=len(resolver.cache),cot_quality_filter=False,
        scope='Full selected JSON scans and path existence; RGB decode only sampled media. Document row index plus exact query/answer checked for all annotations; source original pixels extracted only for samples; author JPEG equivalence not established.')
    write_json(OUT/'audit.json',report);print(report['summary'],flush=True)


def smoke():
    import torch
    from .common import read_jsonl
    from .model import EndpointProcessor,load_teacher
    from .losses import contrastive_loss,positive_mask
    c=load_config('configs/teacher_pilot.yaml');rows=list(read_jsonl(OUT/'samples.jsonl'))
    results=[];proc=EndpointProcessor(c)
    # Same current production budget first. Explicitly separate any smaller smoke budget.
    usable=collections.defaultdict(list)
    for r in rows:
        try:
            for side in ['query','candidate']:
                _,m=proc.endpoint(r[side],'both_cot')
            usable[r['task']].append(r)
        except Exception as e:results.append(dict(stage='production_processor',sample=r['sample_id'],error=repr(e)))
    write_json(OUT/'processor_preflight.json',dict(results=results,usable={k:len(v) for k,v in usable.items()}))
    model=load_teacher(c,Path(c['output'])/'common/initial_adapter',trainable=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
    for task,rr in usable.items():
        # Real 2-pair contrastive pool; this is an input/gradient check, not a benchmark.
        pair=[]
        for r in rr:
            if all(r['candidate']['id']!=s['candidate']['id'] for s in pair):pair.append(r)
            if len(pair)==2:break
        if len(pair)<2:results.append(dict(task=task,error='less_than_two_distinct_candidates'));continue
        for mode in ['no_cot','both_cot']:
            try:
                opt.zero_grad(set_to_none=True);model.train();torch.cuda.reset_peak_memory_stats()
                qi,qmeta=proc.batch([r['query'] for r in pair],mode)
                ci,cmeta=proc.batch([r['candidate'] for r in pair],mode)
                q=model(qi);d=model(ci)
                loss,stats=contrastive_loss(q,d,positive_mask(pair,[r['candidate'] for r in pair]))
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                assert torch.isfinite(loss) and torch.isfinite(norm) and norm>0
                p=next(p for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().max()>0)
                before=p.detach().clone();opt.step();delta=float((p.detach()-before).abs().max());assert delta>0
                media_delta=None
                if mode=='both_cot':
                    visual_input=qi if any(k in qi for k in ['pixel_values','pixel_values_videos']) else ci
                    zeroed={k:([None if x is None else torch.zeros_like(x) for x in v] if k in ['pixel_values','pixel_values_videos'] else v) for k,v in visual_input.items()}
                    model.eval()
                    with torch.no_grad():
                        real=model(visual_input);blank=model(zeroed)
                    media_delta=float((real-blank).abs().max());assert media_delta>1e-6
                results.append(dict(task=task,mode=mode,status='passed',media_embedding_delta=media_delta,loss=float(loss.detach()),gradient_norm=float(norm),parameter_delta=delta,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30,stats=stats,
                    query=[{k:v for k,v in m.items() if k not in ['decoded','retained_cot']} for m in qmeta],
                    candidate=[{k:v for k,v in m.items() if k not in ['decoded','retained_cot']} for m in cmeta],
                    visual_shapes={side:{k:[None if x is None else list(x.shape) for x in v] for k,v in inp.items() if isinstance(v,list)} for side,inp in [('query',qi),('candidate',ci)]}))
                print(task,mode,'passed',float(loss.detach()),flush=True)
            except Exception as e:
                results.append(dict(task=task,mode=mode,status='failed',error=repr(e)));print(task,mode,repr(e),flush=True)
            write_json(OUT/'training_smoke.json',dict(results=results,cot_quality_filter=False,weights_saved=False,scope='Sequential diagnostic updates from common initial adapter, not comparable trained models'))
    opt.zero_grad(set_to_none=True)
    if any(r.get('status')=='failed' or 'error' in r for r in results):
        raise RuntimeError('Some readiness checks failed; inspect training_smoke.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','smoke']);args=p.parse_args()
    audit() if args.action=='audit' else smoke()
