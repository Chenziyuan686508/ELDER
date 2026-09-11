"""Optional small-sample cumulative CoT state export, disabled unless --enable is passed."""
from pathlib import Path
import json
import re
import torch
from .common import parser,load_config,read_jsonl,write_json
from .model import EndpointProcessor,load_teacher


def chunk_boundaries(tokenizer,metadata,input_ids,k):
    if k<1:raise ValueError('K must be positive')
    text=metadata['decoded'];cot=metadata['retained_cot'];start=text.rfind(cot)
    if not cot or start<0:raise ValueError('retained CoT not found')
    tokens=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if tokens['input_ids']!=input_ids:raise ValueError('decoded text did not roundtrip to actual processor token positions')
    tags=[(m.start(),m.end()) for m in re.finditer(r'<[^>]+>',text)]
    body=[]
    for i,(a,b) in enumerate(tokens['offset_mapping']):
        if a<start or b<=a or i==metadata['pooling_position']:continue
        if any(a<y and b>x for x,y in tags):continue
        if text[a:b].strip():body.append((i,a,b))
    if not body:return [],[]
    sentence_ends={m.end()+start for m in re.finditer(r'[.!?](?:\s|$)|\n',cot)}
    choices=[n for n,(_,a,b) in enumerate(body) if any(a<end<=b+1 for end in sentence_ends)]
    choices.append(len(body)-1);ends=[]
    for chunk in range(1,min(k,len(body))+1):
        target=chunk*len(body)/min(k,len(body))-1
        available=[i for i in choices if not ends or i>ends[-1]]
        if not available:break
        ends.append(min(available,key=lambda i:abs(i-target)))
    # Very short CoT can have fewer sentence boundaries than K; never copy states.
    spans=[];previous=0
    for end in ends:
        spans.append(dict(token_start=body[previous][0],token_end=body[end][0],
                          char_start=body[previous][1],char_end=body[end][2]))
        previous=end+1
    return [body[i][0] for i in ends],spans


def encode_with_process(model,processor,endpoint,K=3):
    inputs,meta=processor.batch([endpoint],'both_cot');metadata=meta[0]
    indices,spans=chunk_boundaries(processor.tokenizer,metadata,inputs['input_ids'][0].tolist(),K)
    with torch.no_grad():embedding,h,pool=model(inputs,return_hidden=True)
    states=torch.zeros(K,h.shape[-1],device=h.device,dtype=h.dtype);mask=torch.zeros(K,device=h.device,dtype=torch.bool)
    if indices:states[:len(indices)]=h[0,indices];mask[:len(indices)]=True
    return dict(final_embedding=embedding[0],chunk_end_hidden_states=states,chunk_mask=mask,
                state_norms=states.float().norm(dim=-1),state_cosines=torch.nn.functional.normalize(states.float(),dim=-1) @ torch.nn.functional.normalize(states.float(),dim=-1).T,
                token_spans=spans,original_text_spans=[metadata['decoded'][s['char_start']:s['char_end']] for s in spans],
                layer_id=-1,provenance=endpoint.get('cot_provenance','unknown'),pooling_position=int(pool[0]),
                note='Single complete causal forward; masked padding states are zero, never copied.')


def main():
    p=parser(__doc__);p.add_argument('--enable',action='store_true');p.add_argument('--max-samples',type=int,default=2);args=p.parse_args()
    c=load_config(args.config)
    if not (args.enable or c['process_export']['enabled']):raise SystemExit('Process export disabled; pass --enable for a small diagnostic export')
    if not 1<=args.max_samples<=10:raise ValueError('diagnostic export restricted to 1..10 samples')
    root=Path(c['output']);run=root/'runs'/'both_cot'
    if not (run/'result.json').exists():raise RuntimeError('completed teacher checkpoint required')
    info=json.loads((run/'latest.json').read_text());model=load_teacher(c,run/info['checkpoint']);model.eval();processor=EndpointProcessor(c)
    for n,row in enumerate(read_jsonl(root/'data'/'train.jsonl')):
        if n>=args.max_samples:break
        endpoint=row['query'];result=encode_with_process(model,processor,endpoint,c['process_export']['chunks'])
        original_inputs,metadata=processor.batch([endpoint],'both_cot')
        span=result['token_spans'][0]
        meta=metadata[0];cot_start=meta['decoded'].rfind(meta['retained_cot'])
        altered=dict(endpoint,cot=meta['retained_cot'][:span['char_end']-cot_start]+' A deliberately different later continuation.')
        changed_inputs,_=processor.batch([altered],'both_cot')
        end=span['token_end']
        if not torch.equal(original_inputs['input_ids'][0,:end+1],changed_inputs['input_ids'][0,:end+1]):
            raise ValueError('causality probe changed the prefix tokens')
        with torch.no_grad():_,h,_=model(changed_inputs,return_hidden=True)
        original_state=result['chunk_end_hidden_states'][0].float();changed_state=h[0,end].float()
        cosine=float(torch.nn.functional.cosine_similarity(original_state,changed_state,dim=0))
        error=float((original_state-changed_state).abs().max())
        if cosine<.999:raise RuntimeError(f'causality tolerance failed: cosine={cosine}')
        result['causality_probe']=dict(state_cosine=cosine,max_abs=error,min_cosine=.999,alteration='replace CoT suffix after first block')
        result={k:v.cpu().tolist() if isinstance(v,torch.Tensor) else v for k,v in result.items()}
        write_json(root/'process_export'/f'{n}.json',dict(sample_id=row['sample_id'],split='train_diagnostic',**result))

if __name__=='__main__':main()
