"""Native local Qwen2-VL/VLM2Vec adapter; final valid-token pooling."""
from __future__ import annotations
from pathlib import Path
import re
import torch
from PIL import Image
from torch import nn
from peft import PeftModel,LoraConfig,get_peft_model
from .common import write_json


def processor_for(c):
    from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    from src.model.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
    p=c['model']['name_or_path'];cfg=c['input']
    ip=Qwen2VLImageProcessor.from_pretrained(p,size={'shortest_edge':cfg['min_pixels'],'longest_edge':cfg['max_pixels']})
    ip.min_pixels=cfg['min_pixels'];ip.max_pixels=cfg['max_pixels']
    return Qwen2VLProcessor(image_processor=ip,tokenizer=Qwen2TokenizerFast.from_pretrained(p,padding_side='left'))


class EndpointProcessor:
    def __init__(self,c): self.c=c;self.processor=processor_for(c);self.tokenizer=self.processor.tokenizer
    def endpoint(self,ep,mode):
        if mode not in {'no_cot','both_cot'}: raise ValueError('unknown CoT mode')
        images=[]
        for path in ep['images']:
            with Image.open(path) as im: images.append(im.convert('RGB'))
        frames=[]
        paths=ep['video_frames']
        if paths:
            import numpy as np
            indices=np.linspace(0,len(paths)-1,min(len(paths),self.c['input']['max_video_frames']),dtype=int)
            for i in indices:
                with Image.open(paths[i]) as im: frames.append(im.convert('RGB'))
        prefix='<|vision_start|><|image_pad|><|vision_end|>'*len(images)
        if frames: prefix+='<|vision_start|><|video_pad|><|vision_end|>'
        original=prefix+ep['text']
        def process(text):
            return self.processor(text=[text],images=images or None,videos=[frames] if frames else None,
                                  padding=False,truncation=False,return_tensors='pt')
        base=process(original)
        original_n=base['input_ids'].shape[1];limit=self.c['input']['max_sequence_tokens']
        if original_n==0 or original_n>limit: raise ValueError('original_over_budget_or_empty')
        cot='';full=base;retained=0
        if mode=='both_cot':
            raw=ep['cot'];tokens=self.tokenizer.encode(raw,add_special_tokens=False)
            cap=min(len(tokens),self.c['input']['max_cot_tokens'],max(0,limit-original_n-8))
            while cap>0:
                cot=self.tokenizer.decode(tokens[:cap],skip_special_tokens=False)
                if cap<len(tokens):
                    boundaries=list(re.finditer(r'[.!?](?:\s|$)|\n',cot))
                    if boundaries: cot=cot[:boundaries[-1].end()].strip()
                full=process(original+'\nRetrieval reasoning:\n'+cot)
                if full['input_ids'].shape[1]<=limit: break
                cap-=max(1,full['input_ids'].shape[1]-limit)
            if cap<=0 or not re.sub(r'<[^>]+>','',cot).strip(): raise ValueError('cot_fully_truncated')
            retained=len(self.tokenizer.encode(cot,add_special_tokens=False))
        # Native checkpoint pools last content token, adds no new EOS/embedding tokens.
        ids=full['input_ids'][0]
        for key,tid in [('image_grid_thw',151655),('video_grid_thw',151656)]:
            if key in full:
                expected=int(full[key].prod(dim=-1).sum())//4
                if int((ids==tid).sum())!=expected: raise ValueError('visual_token_mismatch')
        meta=dict(endpoint_id=ep['id'],mode=mode,original_tokens=original_n,total_tokens=len(ids),cot_tokens=retained,
                  cot_truncated=bool(mode=='both_cot' and retained<len(tokens)),pooling_position=len(ids)-1,
                  decoded=self.tokenizer.decode(ids),retained_cot=cot)
        return dict(full),meta
    def batch(self,endpoints,mode):
        processed=[self.endpoint(ep,mode) for ep in endpoints]
        inputs=self.tokenizer.pad([{'input_ids':x[0]['input_ids'][0].tolist()} for x in processed],padding=True,return_tensors='pt')
        for key in ['pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw']:
            chunks=[x[0].get(key) for x in processed]
            if any(x is not None for x in chunks): inputs[key]=chunks
        return dict(inputs),[x[1] for x in processed]


class Teacher(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        base=encoder.get_base_model() if isinstance(encoder,PeftModel) else encoder
        # No LM objective: bypass vocabulary projection but preserve its tied parameter/state key.
        base.lm_head.forward=lambda hidden: hidden[..., :1]
    def forward(self,inputs,return_hidden=False):
        device=next(self.parameters()).device
        def move(v):
            if isinstance(v,list): return [move(x) for x in v]
            return v.to(device) if isinstance(v,torch.Tensor) else v
        inputs={k:move(v) for k,v in inputs.items()}
        # Force a fresh multimodal rope computation on every independent encoding.
        inputs['cache_position']=torch.arange(inputs['input_ids'].shape[1],device=inputs['input_ids'].device)
        base=self.encoder.get_base_model() if isinstance(self.encoder,PeftModel) else self.encoder
        def grids(key):
            values=[v for v in inputs.get(key,[]) if v is not None]
            return torch.cat(values) if values else None
        inputs['position_ids'],_=base.get_rope_index(inputs['input_ids'],grids('image_grid_thw'),grids('video_grid_thw'),inputs['attention_mask'])
        output=self.encoder(**inputs,use_cache=False,output_hidden_states=True,return_dict=True)
        h=output.hidden_states[-1];mask=inputs['attention_mask']
        positions=torch.arange(mask.shape[1],device=mask.device).expand_as(mask).masked_fill(~mask.bool(),-1).max(1).values
        if (positions<0).any(): raise ValueError('empty pooling row')
        embedding=torch.nn.functional.normalize(h[torch.arange(len(h),device=h.device),positions].float(),dim=-1)
        return (embedding,h,positions) if return_hidden else embedding


def load_backbone(path):
    from src.model.vlm_backbone.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    m=Qwen2VLForConditionalGeneration.from_pretrained(str(path),torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',local_files_only=True)
    if m.config.model_type!='qwen2_vl' or m.config.hidden_size!=1536: raise ValueError('unexpected backbone')
    return m


def load_teacher(c,adapter=None,trainable=False,released=False):
    root=Path(c['output'])
    path=c['model']['backbone'] if released else root/'common'/'merged_base'
    base=load_backbone(path)
    if released: base=PeftModel.from_pretrained(base,c['model']['name_or_path'],is_trainable=False)
    elif adapter:
        base=PeftModel.from_pretrained(base,str(adapter),is_trainable=trainable)
        for config in base.peft_config.values():config.base_model_name_or_path=str(path)
    if trainable:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        base.enable_input_require_grads()
    return Teacher(base).cuda()


def initialize(c):
    root=Path(c['output'])/'common';root.mkdir(parents=True,exist_ok=True)
    if (root/'initialization.json').exists():
        import json
        saved=json.loads((root/'initialization.json').read_text())
        if saved['model']!=c['model'] or saved['seed']!=c['experiment']['seed']:
            raise ValueError('existing common initialization differs from requested configuration')
        return
    torch.manual_seed(c['experiment']['seed'])
    model=load_teacher(c,released=True);model.eval();proc=EndpointProcessor(c)
    ep=dict(id='init',text='Represent this retrieval endpoint.',images=[],video_frames=[],cot='A short representation of retrieval semantics.')
    inp,_=proc.batch([ep],'no_cot')
    with torch.no_grad(): before=model(inp).cpu()
    merged=model.encoder.merge_and_unload(safe_merge=True);model=Teacher(merged)
    with torch.no_grad(): after=model(inp).cpu()
    error=float((before-after).abs().max())
    if error>0.01: raise RuntimeError(f'DoRA merge drift {error}')
    merged.save_pretrained(root/'merged_base',safe_serialization=True)
    proc.processor.save_pretrained(root/'merged_base')
    cfg=c['model'];targets=[n for n,m in merged.named_modules() if isinstance(m,nn.Linear) and n.startswith('model.layers.') and n.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}]
    if len(targets)!=28*7: raise RuntimeError('unexpected language LoRA scope')
    adapted=get_peft_model(merged,LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],target_modules=targets,bias='none'))
    names={n:p.numel() for n,p in adapted.named_parameters() if p.requires_grad}
    if any('visual' in n for n in names): raise RuntimeError('visual module trainable')
    for config in adapted.peft_config.values():config.base_model_name_or_path=str(root/'merged_base')
    adapted.save_pretrained(root/'initial_adapter');proc.processor.save_pretrained(root/'initial_adapter')
    write_json(root/'initialization.json',dict(model=c['model'],trainable_parameters=sum(names.values()),parameters=names,target_modules=targets,
        visual_frozen=all(not p.requires_grad for n,p in adapted.named_parameters() if 'visual' in n),
        merge_max_abs=error,pooling='last valid content token; no extra special token',seed=c['experiment']['seed']))
