"""Final real-model endpoint isolation, padding and checkpoint dependency checks."""
from pathlib import Path
import json
import torch
from .common import parser,load_config,read_jsonl,write_json
from .model import EndpointProcessor,load_teacher


def main():
    args=parser(__doc__).parse_args();c=load_config(args.config);root=Path(c['output']);run=root/'runs/both_cot'
    info=json.loads((run/'latest.json').read_text());checkpoint=run/info['checkpoint']
    cfg=json.loads((checkpoint/'adapter_config.json').read_text())
    assert Path(cfg['base_model_name_or_path']).resolve()==(root/'common/merged_base').resolve()
    model=load_teacher(c,checkpoint);model.eval();p=EndpointProcessor(c);row=next(read_jsonl(root/'data/train.jsonl'))
    q=row['query'];d=row['candidate'];d_changed=dict(d,cot='This is a deliberately different description focusing on different semantic content.')
    def encode(eps,mode):
        inputs,meta=p.batch(eps,mode)
        with torch.no_grad():out=model(inputs)
        return out,meta
    dc,_=encode([d],'both_cot');changed,_=encode([d_changed],'both_cot')
    delta=float((dc-changed).abs().max());assert delta>1e-6
    encode([q],'both_cot');d_after_q,_=encode([d],'both_cot')
    encode([dict(q,cot='A different query explanation.')],'both_cot');d_after_other_q,_=encode([d],'both_cot')
    independent=float((d_after_q-d_after_other_q).abs().max());assert independent==0
    left,_=encode([q,d],'both_cot');p.tokenizer.padding_side='right';right,_=encode([q,d],'both_cot')
    pad_error=float((left-right).abs().max());pad_cos=torch.nn.functional.cosine_similarity(left,right,dim=-1)
    if pad_error>.005 or float(pad_cos.min())<.999:raise RuntimeError('padding changes retrieval semantics beyond bf16 tolerance')
    for ep in [q,d]:
        _,meta=encode([ep],'no_cot')
        assert ep['cot'] not in meta[0]['decoded']
    report=dict(status='passed',candidate_cot_embedding_change=delta,candidate_after_different_query_max_abs=independent,
        left_right_padding_max_abs=pad_error,left_right_padding_cosines=pad_cos.tolist(),padding_max_abs_tolerance=.005,
        adapter_points_to_merged_retrieval_base=True,model_mode='eval',checkpoint=str(checkpoint))
    write_json(root/'final_model_checks.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
