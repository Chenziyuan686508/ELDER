"""Generate held-out CoT from endpoint-only prompts; never consume stored training CoT."""
import json
import time
from pathlib import Path
from .common import parser,load_config,read_jsonl,write_json,digest,file_digest
from .data import content_key,endpoint_prompt


def main():
    p=parser(__doc__);p.add_argument('--max-items',type=int);p.add_argument('--text-batch-size',type=int,default=64);p.add_argument('--image-batch-size',type=int,default=8)
    args=p.parse_args();c=load_config(args.config);out=Path(c['output']);cfg=c['evaluation'];model_path=Path(cfg['generator_model'])
    if cfg['allow_reference_answers']: raise ValueError('normal evaluation cannot use reference answers')
    if min(args.text_batch_size,args.image_batch_size)<1:raise ValueError('batch sizes must be positive')
    contract=dict(split='validation',model=str(model_path),
        model_files={p.name:file_digest(p) for p in model_path.glob('*') if p.suffix in {'.json','.jinja','.safetensors'}},
        prompt_version='endpoint_only_natural_cot_v1',temperature=cfg['generator_temperature'],max_new_tokens=cfg['generator_max_new_tokens'],
        max_pixels=c['input']['max_pixels'],max_video_frames=c['input']['max_video_frames'],reference_answers=False,
        text_batch_size=args.text_batch_size,image_batch_size=args.image_batch_size)
    version=digest(contract);cache_dir=out/'eval_cot'/version;cache_dir.mkdir(parents=True,exist_ok=True)
    write_json(cache_dir/'manifest.json',contract);write_json(out/'eval_cot'/'active.json',dict(version=version))
    jobs={}
    for row in read_jsonl(out/'data'/'validation.jsonl'):
        ep=row['query'];jobs[('query',row['task'],ep['id'])]=content_key(ep)
    for task in c['data']['selected_tasks']:
        for ep in read_jsonl(out/'data'/f'{task}.candidates.jsonl'):
            jobs[('candidate',task,ep['id'])]=content_key(ep)
    pending=[];completed=0
    for (role,task,eid),ep in sorted(jobs.items()):
        key=digest([version,role,task,eid,ep]);path=cache_dir/(key+'.json')
        if path.exists() and json.loads(path.read_text()).get('status')=='ok':completed+=1;continue
        pending.append(dict(role=role,task=task,endpoint_id=eid,endpoint=ep,key=key,path=path))
    if not pending:print('cache complete',completed);return
    if args.max_items is not None:pending=pending[:args.max_items]
    from scripts.elder.generate_cot import GlmGenerator
    import torch
    generator=GlmGenerator(model_path=model_path,attn_implementation='sdpa',max_new_tokens=cfg['generator_max_new_tokens'],
        temperature=cfg['generator_temperature'],top_p=1,seed=c['experiment']['seed'])
    generator.processor.tokenizer.padding_side='left'
    ip=generator.processor.image_processor
    if hasattr(ip,'max_pixels'):ip.max_pixels=c['input']['max_pixels']
    if hasattr(ip,'size') and isinstance(ip.size,dict) and 'longest_edge' in ip.size:
        ip.size['longest_edge']=c['input']['max_pixels']
    write_json(cache_dir/'resolved_processor.json',ip.to_dict())
    start=time.monotonic();failures=0
    def generate(batch):
        conversations=[];prompts=[]
        for job in batch:
            ep=job['endpoint']
            if ep['video_frames']:raise NotImplementedError('video is outside this two-image-task pilot')
            prompt=endpoint_prompt(ep,job['role']);prompts.append(prompt)
            content=[{'type':'image','url':path} for path in ep['images']]+[{'type':'text','text':prompt}]
            conversations.append([{'role':'user','content':content}])
        inputs=generator.processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,
            padding=True,return_dict=True,return_tensors='pt').to(generator.device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            outputs=generator.model.generate(**inputs,max_new_tokens=cfg['generator_max_new_tokens'],do_sample=False,
                pad_token_id=generator.processor.tokenizer.pad_token_id)
        generated=outputs[:,inputs['input_ids'].shape[1]:]
        texts=generator.processor.batch_decode(generated,skip_special_tokens=True)
        for job,prompt,cot,tokens in zip(batch,prompts,texts,generated):
            if not cot.strip():raise ValueError('empty generator response')
            n=int((tokens!=generator.processor.tokenizer.pad_token_id).sum())
            result=dict(status='ok',role=job['role'],task=job['task'],endpoint_id=job['endpoint_id'],endpoint=job['endpoint'],
                cot=cot.strip(),cot_provenance='independent_generator',prompt=prompt,version=version,
                generated_tokens=n,hit_generation_cap=n>=cfg['generator_max_new_tokens'],actual_batch_size=len(batch),
                peak_memory_bytes=torch.cuda.max_memory_allocated())
            write_json(job['path'],result)
    while pending:
        first=pending[0];visual=bool(first['endpoint']['images']);size=args.image_batch_size if visual else args.text_batch_size
        batch=[]
        while pending and len(batch)<size and bool(pending[0]['endpoint']['images'])==visual:batch.append(pending.pop(0))
        try:generate(batch)
        except Exception as exc:
            print('batch failure, retry individually',type(exc).__name__,str(exc),flush=True)
            torch.cuda.empty_cache()
            for job in batch:
                for attempt in range(3):
                    try:generate([job]);break
                    except Exception as err:
                        error=f'{type(err).__name__}: {err}';torch.cuda.empty_cache()
                else:
                    failures+=1;write_json(job['path'],dict(status='failed',error=error,endpoint_id=job['endpoint_id']))
        completed+=sum(json.loads(job['path'].read_text()).get('status')=='ok' for job in batch)
        progress=dict(required=len(jobs),complete=completed,failed_this_run=failures,seconds=time.monotonic()-start)
        write_json(cache_dir/'progress.json',progress);print(json.dumps(progress),flush=True)
    if completed<len(jobs):raise SystemExit('Normal evaluation CoT cache incomplete; partial cache retained for resume')

if __name__=='__main__':main()
