"""Verify both modes fit before pilot; reject original/CoT budget failures jointly."""
from collections import Counter
from pathlib import Path
import json
from .common import parser,load_config,read_jsonl,write_json,write_jsonl
from .model import EndpointProcessor


def main():
    args=parser(__doc__).parse_args();c=load_config(args.config);root=Path(c['output']);processor=EndpointProcessor(c)
    if any((root/'runs').glob('*/latest.json')):raise RuntimeError('Do not change input manifest after pilot starts')
    rows=list(read_jsonl(root/'data'/'train.jsonl'));failures=[];stats=Counter();cache={}
    for i,row in enumerate(rows):
        for side in ['query','candidate']:
            ep=row[side]
            key=(row['task'],side,ep['id'])
            if key in cache:
                if cache[key]: failures.append(dict(sample_id=row['sample_id'],side=side,error=cache[key]))
                continue
            try:
                for mode in ['no_cot','both_cot']:
                    _,meta=processor.endpoint(ep,mode)
                    stats[mode+'_tokens']+=meta['total_tokens'];stats[mode+'_cot_tokens']+=meta['cot_tokens']
                    stats[mode+'_truncated']+=int(meta['cot_truncated'])
                    stats['max_sequence_tokens']=max(stats['max_sequence_tokens'],meta['total_tokens'])
                cache[key]=None
            except ValueError as e:
                cache[key]=str(e);failures.append(dict(sample_id=row['sample_id'],side=side,error=str(e)))
        if (i+1)%200==0:print('preflight',i+1,dict(stats),flush=True)
    bad={f['sample_id'] for f in failures};filtered=[r for r in rows if r['sample_id'] not in bad]
    plan=json.loads((root/'data'/'batch_plan.json').read_text())
    for b in plan:b['sample_ids']=[i for i in b['sample_ids'] if i not in bad]
    if any(len(b['sample_ids'])<2 for b in plan):raise RuntimeError('input filter leaves an invalid contrastive batch; rebuild plan')
    if bad:
        write_jsonl(root/'data'/'train.jsonl',filtered);write_json(root/'data'/'batch_plan.json',plan)
    write_json(root/'input_budget_audit.json',dict(status='passed',scanned_train_pairs=len(rows),retained_train_pairs=len(filtered),failures=failures,statistics=stats))
    print('input preflight passed',len(filtered),flush=True)

if __name__=='__main__':main()
