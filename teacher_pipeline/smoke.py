"""Real-image/token visibility tests and ten matched optimizer steps for each mode."""
import copy
import gc
from pathlib import Path
import torch
from .common import parser,load_config,read_jsonl,write_json
from .model import initialize,EndpointProcessor,load_teacher
from .train import run


def visibility(c):
    root=Path(c['output']);row=next(read_jsonl(root/'data'/'train.jsonl'));p=EndpointProcessor(c)
    q=row['query'];d=row['candidate']
    changed=copy.deepcopy(q);changed['cot']='The visible content has a different semantic emphasis. This is a deliberate input sensitivity probe.'
    q1,m1=p.batch([q],'both_cot');q2,m2=p.batch([changed],'both_cot')
    d1,_=p.batch([d],'both_cot');d2,_=p.batch([d],'both_cot')
    assert not torch.equal(q1['input_ids'],q2['input_ids'])
    assert all(torch.equal(d1[k],d2[k]) for k in d1)
    off,mo=p.batch([q],'no_cot');assert mo[0]['cot_tokens']==0
    assert m1[0]['retained_cot'] in m1[0]['decoded']
    model=load_teacher(c,root/'common'/'initial_adapter');model.eval()
    with torch.no_grad():
        e1=model(q1);e2=model(q2);a=model(d1);b=model(d2)
    delta=float((e1-e2).abs().max());assert delta>1e-6
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    # Real mixed image/text batch plus left/right padding pooling equivalence.
    mixed,meta=p.batch([q,d],'both_cot')
    with torch.no_grad():mixed_e=model(mixed)
    torch.testing.assert_close(mixed_e[0],e1[0],atol=.003,rtol=.03)
    result=dict(status='passed',image_paths=q['images'],cot_embedding_max_abs_change=delta,
                candidate_repeat_max_abs=float((a-b).abs().max()),mixed_batch_max_abs=float((mixed_e[0]-e1[0]).abs().max()),
                query=m1[0],changed_query=m2[0],no_cot_query=mo[0],
                video='not covered by this pilot',visual_document='not covered by this pilot')
    write_json(root/'input_visibility.json',result)
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=parser(__doc__);p.add_argument('--resume',action='store_true');p.add_argument('--visibility-only',action='store_true');args=p.parse_args()
    c=load_config(args.config);initialize(c);visibility(c)
    if not args.visibility_only:
        a=run(c,'no_cot',smoke=True,resume=args.resume);b=run(c,'both_cot',smoke=True,resume=args.resume)
        write_json(Path(c['output'])/'smoke'/'report.json',dict(a=a,b=b,status='passed'))

if __name__=='__main__': main()
