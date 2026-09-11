"""Full held-out candidate-corpus evaluation with group-paired bootstrap."""
from pathlib import Path
import gc
import json
import time
import numpy as np
import torch
from .common import parser,load_config,read_jsonl,write_json,digest,file_digest
from .model import EndpointProcessor,load_teacher
from .data import content_key


def retrieval_metrics(scores,positive_ids,candidate_ids):
    index={cid:i for i,cid in enumerate(candidate_ids)}
    values=[]
    for row,positives in zip(scores,positive_ids):
        relevant={index[p] for p in positives if p in index}
        if not relevant:raise ValueError('query has no relevant candidate in evaluation corpus')
        order=np.argsort(-row,kind='stable')
        rank=min(i+1 for i,j in enumerate(order) if j in relevant)
        values.append(dict(hit1=float(rank<=1),recall1=len(set(order[:1])&relevant)/len(relevant),
            recall5=len(set(order[:5])&relevant)/len(relevant),recall10=len(set(order[:10])&relevant)/len(relevant),rank=rank))
    return values


def paired_bootstrap(a,b,groups,seed=42,iterations=2000):
    unique=sorted(set(groups));values={g:[] for g in unique}
    for x,y,g in zip(a,b,groups):values[g].append(y-x)
    rng=np.random.default_rng(seed);deltas=[]
    for _ in range(iterations):
        sample=rng.choice(unique,len(unique),replace=True)
        deltas.append(np.mean([v for g in sample for v in values[g]]))
    return dict(mean=float(np.mean(np.array(b)-np.array(a))),ci95=np.percentile(deltas,[2.5,97.5]).tolist(),unit='source_group',iterations=iterations,seed=seed)


def main():
    p=parser(__doc__);p.add_argument('--suite',default='base,a,b,b_off');args=p.parse_args();c=load_config(args.config)
    root=Path(c['output']);proc=EndpointProcessor(c);queries=list(read_jsonl(root/'data'/'validation.jsonl'));all_results={}
    for group in args.suite.split(','):
        if group not in {'base','a','b','b_off'}:raise ValueError('unknown evaluation group')
        mode='both_cot' if group=='b' else 'no_cot';adapter=None
        if group!='base':
            run=root/'runs'/('no_cot' if group=='a' else 'both_cot')
            info=json.loads((run/'latest.json').read_text());adapter=run/info['checkpoint']
            if not (run/'result.json').exists():raise RuntimeError('pilot training/reload not completed')
        cot_cache={};cot_version=None
        if mode=='both_cot':
            cot_version=json.loads((root/'eval_cot'/'active.json').read_text())['version']
            for pth in (root/'eval_cot'/cot_version).glob('*.json'):
                item=json.loads(pth.read_text())
                if item.get('status')=='ok':cot_cache[(item['role'],item['task'],item['endpoint_id'])]=item
        model=load_teacher(c,adapter,released=group=='base');model.eval();task_results={}
        model_hash=file_digest(Path(c['model']['name_or_path'])/'adapter_model.bin') if group=='base' else file_digest(adapter/'adapter_model.safetensors')
        for task in c['data']['selected_tasks']:
            qs=[r for r in queries if r['task']==task];ds=list(read_jsonl(root/'data'/f'{task}.candidates.jsonl'))
            def encode(endpoints,role):
                cache_key=digest([model_hash,c['model'],c['input'],mode,cot_version,task,role,endpoints,file_digest(Path(__file__).with_name('model.py'))])
                cache=root/'embeddings'/cache_key
                cache.mkdir(parents=True,exist_ok=True)
                if (cache/'vectors.npz').exists() and (cache/'metadata.json').exists():
                    return np.load(cache/'vectors.npz')['embeddings'],json.loads((cache/'metadata.json').read_text())
                embeddings=[];token_count=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
                for i in range(0,len(endpoints),2):
                    batch=[]
                    for ep in endpoints[i:i+2]:
                        ep=dict(ep)
                        if mode=='both_cot':
                            item=cot_cache.get((role,task,ep['id']))
                            if item is not None and item['endpoint']!=content_key(ep):raise RuntimeError('stale endpoint CoT cache')
                            if item is None:raise RuntimeError(f'missing independent CoT for {role}/{task}/{ep["id"]}')
                            ep['cot']=item['cot']
                        batch.append(ep)
                    inp,metadata=proc.batch(batch,mode)
                    token_count+=sum(m['total_tokens'] for m in metadata)
                    with torch.no_grad(): embeddings.append(model(inp).cpu())
                array=torch.cat(embeddings).numpy()
                stats=dict(seconds=time.monotonic()-started,endpoints=len(endpoints),tokens=token_count,peak_memory_bytes=torch.cuda.max_memory_allocated(),batch_size=2,cache_key=cache_key)
                np.savez(cache/'vectors.npz',embeddings=array)
                write_json(cache/'metadata.json',stats)
                return array,stats
            qe,qcost=encode([r['query'] for r in qs],'query');de,dcost=encode(ds,'candidate');scores=qe@de.T
            per_query=retrieval_metrics(scores,[r['positive_candidate_ids'] for r in qs],[d['id'] for d in ds])
            summary={k:float(np.mean([r[k] for r in per_query])) for k in ['hit1','recall1','recall5','recall10']}
            result=dict(metrics=summary,query_ids=[r['query']['id'] for r in qs],groups=[r['source_group_id'] for r in qs],per_query=per_query,
                candidates=len(ds),queries=len(qs),encoding_cost=dict(query=qcost,candidate=dcost),protocol='fixed grouped holdout, full held-out filtered candidate corpus; not official MMEB-V2',cot_version=cot_version)
            result['error_examples']=[dict(sample_id=qs[i]['sample_id'],query=qs[i]['query']['text'],query_images=qs[i]['query']['images'],predicted_candidate=ds[int(np.argmax(scores[i]))]['text'],known_positive=qs[i]['candidate']['text'],rank=per_query[i]['rank']) for i in range(len(qs)) if per_query[i]['hit1']==0][:10]
            task_results[task]=result
            print(group,task,summary,flush=True)
        write_json(root/'evaluation'/f'{group}.json',task_results);all_results[group]=task_results
        del model;gc.collect();torch.cuda.empty_cache()
    for name in ['a','b']:
        pth=root/'evaluation'/f'{name}.json'
        if pth.exists():all_results[name]=json.loads(pth.read_text())
    if 'a' in all_results and 'b' in all_results:
        intervals={}
        for task in c['data']['selected_tasks']:
            a=all_results['a'][task];b=all_results['b'][task]
            if a['query_ids']!=b['query_ids']:raise ValueError('unpaired query sets')
            intervals[task]=paired_bootstrap([r['hit1'] for r in a['per_query']],[r['hit1'] for r in b['per_query']],a['groups'])
        write_json(root/'evaluation'/'b_minus_a.json',intervals)

if __name__=='__main__':main()
