"""Train one bounded matched A/B pilot with exact local GradCache and resumable checkpoints."""
from pathlib import Path
import gc
import json
import math
import random
import time
import torch
import yaml
from .common import parser,load_config,read_jsonl,write_json,write_jsonl,digest,file_digest
from .model import EndpointProcessor,initialize,load_teacher
from .losses import gradcache_backward,positive_mask


def chunks(endpoints,processor,mode,size):
    inputs=[];metadata=[]
    for i in range(0,len(endpoints),size):
        batch,meta=processor.batch(endpoints[i:i+size],mode)
        inputs.append(batch);metadata.extend(meta)
    return inputs,metadata


def run(c,mode,smoke=False,resume=False,stop_after_steps=None):
    initialize(c);root=Path(c['output']);out=root/('smoke' if smoke else 'runs')/mode;out.mkdir(parents=True,exist_ok=True)
    if not smoke and not (root/'smoke'/'report.json').exists(): raise RuntimeError('run successful smoke first')
    if not smoke and not (root/'input_budget_audit.json').exists(): raise RuntimeError('run shared input preflight first')
    rows={r['sample_id']:r for r in read_jsonl(root/'data'/'train.jsonl')}
    plan=json.loads((root/'data'/'batch_plan.json').read_text())
    if smoke:
        # 10 optimizer steps on real task-homogeneous pairs; smaller pool explicitly reported.
        plan=[dict(task=b['task'],sample_ids=b['sample_ids'][:4]) for b in plan[:10]]
    if len(plan)<10 and smoke: raise RuntimeError('need at least 10 smoke steps')
    for batch in plan:
        if len(batch['sample_ids'])!=len(set(batch['sample_ids'])):
            raise ValueError('duplicate query row in batch plan')
        if any(rows[sample]['task']!=batch['task'] for sample in batch['sample_ids']):
            raise ValueError('cross-task negatives in supposedly homogeneous batch')
    fingerprint=digest([c,plan,file_digest(root/'common'/'initial_adapter'/'adapter_model.safetensors'),file_digest(root/'data'/'train.jsonl')])
    latest=out/'latest.json';start=0
    if latest.exists() and not resume: raise RuntimeError(f'existing run {out}: use --resume')
    adapter=root/'common'/'initial_adapter'
    if resume and latest.exists():
        info=json.loads(latest.read_text())
        if info['fingerprint']!=fingerprint: raise ValueError('resume configuration/data mismatch')
        adapter=out/info['checkpoint'];start=info['step']
    seed=c['experiment']['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
    model=load_teacher(c,adapter,trainable=True);model.train();processor=EndpointProcessor(c)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay'])
    warmup=max(1,math.ceil(len(plan)*c['training']['warmup_ratio']))
    schedule=lambda step: (step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,len(plan)-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    if start:
        state=torch.load(adapter/'training_state.pt',map_location='cpu',weights_only=False)
        opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
    resolved=dict(c,mode=mode,smoke=smoke,actual_optimizer_steps=len(plan),actual_contrastive_pairs=4 if smoke else c['training']['contrastive_pool_target_pairs'],fingerprint=fingerprint)
    (out/'resolved.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False))
    def save(step):
        checkpoint=out/f'checkpoint-{step}'
        model.encoder.save_pretrained(checkpoint);processor.processor.save_pretrained(checkpoint)
        torch.save(dict(optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()),checkpoint/'training_state.pt')
        write_json(latest,dict(step=step,checkpoint=checkpoint.name,fingerprint=fingerprint))
    last_inputs=None
    for step in range(start,len(plan)):
        batch=[rows[i] for i in plan[step]['sample_ids']]
        candidates=list({r['candidate']['id']:r['candidate'] for r in batch}.values())
        q_chunks,q_meta=chunks([r['query'] for r in batch],processor,mode,c['training']['microbatch_pairs'])
        d_chunks,d_meta=chunks(candidates,processor,mode,c['training']['microbatch_pairs'])
        positives=positive_mask(batch,candidates,'cuda')
        opt.zero_grad(set_to_none=True);begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
        before=[p.detach().clone() for p in params] if step==start else None
        loss,stats=gradcache_backward(model,q_chunks,d_chunks,positives,c['training']['temperature'])
        norm=torch.nn.utils.clip_grad_norm_(params,c['training']['gradient_clip_norm'])
        if not torch.isfinite(loss) or not torch.isfinite(norm) or norm<=0: raise RuntimeError('nonfinite loss/gradient or zero gradient')
        opt.step();scheduler.step()
        if before is not None and not any(not torch.equal(p.detach(),old) for p,old in zip(params,before)):
            raise RuntimeError('no trainable parameter updated')
        del before
        stats.update(step=step+1,mode=mode,task=plan[step]['task'],loss=float(loss),grad_norm=float(norm),
            seconds=time.monotonic()-begin,peak_memory_bytes=torch.cuda.max_memory_allocated(),
            input_tokens=sum(x['total_tokens'] for x in q_meta+d_meta),cot_tokens=sum(x['cot_tokens'] for x in q_meta+d_meta),
            cot_truncated=sum(x['cot_truncated'] for x in q_meta+d_meta),optimizer_pairs=len(batch))
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
        print(json.dumps(stats),flush=True)
        if step==0: write_json(out/'actual_inputs.json',dict(query=q_meta[0],candidate=d_meta[0]))
        if (step+1)%c['training']['save_every']==0 or step+1==len(plan):save(step+1)
        last_inputs=q_chunks[0]
        if stop_after_steps is not None and step+1>=stop_after_steps and step+1<len(plan):
            save(step+1)
            write_json(out/'pause.json',dict(status='paused',completed_steps=step+1,total_steps=len(plan)))
            return dict(status='paused',completed_steps=step+1)
    if last_inputs is None:
        r=rows[plan[-1]['sample_ids'][0]];last_inputs=processor.batch([r['query']],mode)[0]
    model.eval()
    with torch.no_grad():expected=model(last_inputs).cpu()
    del model,opt,scheduler,params;gc.collect();torch.cuda.empty_cache()
    info=json.loads(latest.read_text());reloaded=load_teacher(c,out/info['checkpoint']);reloaded.eval()
    with torch.no_grad():actual=reloaded(last_inputs).cpu()
    error=float((expected-actual).abs().max())
    if error>2e-4:raise RuntimeError(f'checkpoint reload mismatch {error}')
    result=dict(mode=mode,optimizer_steps=len(plan),reload_max_abs=error,reload_atol=2e-4,status='passed')
    write_json(out/'result.json',result)
    del reloaded;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=parser(__doc__);p.add_argument('--mode',required=True,choices=['no_cot','both_cot']);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-steps',type=int,help='Save at this global step, for bounded execution/resume verification')
    args=p.parse_args();run(load_config(args.config),args.mode,resume=args.resume,stop_after_steps=args.stop_after_steps)

if __name__=='__main__': main()
