import json
import pytest
import torch
from teacher_pipeline.common import iter_json_array
from teacher_pipeline.data import canonical_id,clean_text,endpoint_prompt,MediaResolver,normalize
from teacher_pipeline.losses import contrastive_loss,gradcache_backward


def test_stream(tmp_path):
    rows=[{'x':'你好'*i,'y':[1,2]} for i in range(20)]
    p=tmp_path/'x.json';p.write_text(json.dumps(rows,ensure_ascii=False))
    assert list(iter_json_array(p,7))==rows
    for text in ['[{"x":1},]','[{"x":1}','[]junk','[{"x":1} {"y":2}]']:
        p.write_text(text)
        with pytest.raises((ValueError,json.JSONDecodeError)): list(iter_json_array(p,7))


def test_identity_and_generator_input():
    ep=dict(text='dog',images=[],video_frames=[],cot='one',id='id')
    assert canonical_id('x',ep)==canonical_id('x',dict(ep,cot='two'))
    assert canonical_id('x',ep)!=canonical_id('y',ep)
    assert 'secret_cot' not in endpoint_prompt(dict(ep,cot='secret_cot'),'candidate')
    assert clean_text('<|image_1|> cat <video>')=='cat'


def test_blank_video_role(tmp_path):
    r=MediaResolver({})
    raw={'dataset_name':'llavahound_video_retrieval','qry':{'video':['images/blank.jpg'],'conversations':[{'from':'human','value':'secret_question'},{'from':'gpt','value':'caption'}]},'pos':{'conversations':[{'from':'human','value':'represent video'},{'from':'gpt','value':'secret_answer'}]},'query_cot':'q','pos_cot':'d'}
    row=normalize(raw,'video',0,r)
    assert row['query']['text']=='caption'
    assert row['candidate']['text']=='represent video'
    assert row['query']['video_frames']==[]


def test_loss_matches_ce_and_masks():
    torch.manual_seed(1);q=torch.randn(4,5,requires_grad=True);d=torch.randn(4,5,requires_grad=True)
    scores=q@d.T/0.2
    actual,_=contrastive_loss(q,d,torch.eye(4,dtype=torch.bool),0.2)
    expected=(torch.nn.functional.cross_entropy(scores,torch.arange(4))+torch.nn.functional.cross_entropy(scores.T,torch.arange(4)))/2
    torch.testing.assert_close(actual,expected)
    p=torch.tensor([[1,1,0,0],[0,0,1,0],[0,0,0,0],[1,0,0,0]],dtype=torch.bool)
    loss,stats=contrastive_loss(q,d,p,0.2)
    loss.backward();assert torch.isfinite(q.grad).all()
    assert stats['skipped_queries']==1 and stats['skipped_candidates']==1
    with pytest.raises(ValueError): contrastive_loss(q,d,torch.ones_like(p))


def test_gradcache_rng_and_gradients():
    torch.manual_seed(2)
    model=torch.nn.Sequential(torch.nn.Linear(5,7),torch.nn.Dropout(.2),torch.nn.Linear(7,4))
    q=[torch.randn(2,5),torch.randn(2,5)];d=[torch.randn(2,5),torch.randn(1,5)]
    p=torch.tensor([[1,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=torch.bool)
    def encode(x): return torch.nn.functional.normalize(model(x),dim=-1)
    state=torch.get_rng_state()
    loss,_=contrastive_loss(torch.cat([encode(x) for x in q]),torch.cat([encode(x) for x in d]),p,.2)
    loss.backward();grads=[x.grad.clone() for x in model.parameters()];after=torch.get_rng_state()
    model.zero_grad();torch.set_rng_state(state)
    cached,_=gradcache_backward(encode,q,d,p,.2)
    torch.testing.assert_close(cached,loss)
    for g,param in zip(grads,model.parameters()): torch.testing.assert_close(param.grad,g,atol=1e-6,rtol=1e-5)
    assert torch.equal(torch.get_rng_state(),after)


def test_full_corpus_multipositive_metrics():
    import numpy as np
    from teacher_pipeline.evaluate import retrieval_metrics,paired_bootstrap
    rows=retrieval_metrics(np.array([[.8,.9,.1],[.1,.2,.3]]),[['a','b'],['a']],['a','b','c'])
    assert rows[0]['hit1']==1 and rows[0]['recall1']==.5
    assert rows[1]['hit1']==0 and rows[1]['rank']==3
    ci=paired_bootstrap([0,0],[1,1],['g','g'])
    assert ci['ci95']==[1.,1.]


def test_answer_canonicalization():
    ep=dict(text='  A Dog. ',images=[],video_frames=[])
    assert canonical_id('Visual7W',ep)==canonical_id('Visual7W',dict(ep,text='a dog'))
    assert canonical_id('MSCOCO_i2t',ep)!=canonical_id('MSCOCO_i2t',dict(ep,text='a dog'))


def test_real_processor_endpoint_isolation_and_budget():
    from pathlib import Path
    from teacher_pipeline.common import load_config,read_jsonl
    from teacher_pipeline.model import EndpointProcessor
    c=load_config('configs/teacher_pilot.yaml')
    path=Path(c['output'])/'data'/'train.jsonl'
    if not path.exists():pytest.skip('local pilot manifest unavailable')
    row=next(read_jsonl(path));proc=EndpointProcessor(c)
    for side,other in [('query','candidate'),('candidate','query')]:
        ep=row[side];before,meta=proc.endpoint(ep,'both_cot')
        changed,_=proc.endpoint(dict(ep,cot='A deliberately distinct sentence about retrieval semantics.'),'both_cot')
        assert not torch.equal(before['input_ids'],changed['input_ids'])
        frozen1,_=proc.endpoint(row[other],'both_cot');frozen2,_=proc.endpoint(row[other],'both_cot')
        assert all(torch.equal(frozen1[k],frozen2[k]) for k in frozen1)
        off,om=proc.endpoint(ep,'no_cot');assert om['cot_tokens']==0
        assert meta['retained_cot'] in meta['decoded']
    ep=row['query'];_,base_meta=proc.endpoint(ep,'no_cot')
    original_budget=c['input']['max_sequence_tokens']
    c['input']['max_sequence_tokens']=base_meta['total_tokens']+20
    _,m=proc.endpoint(dict(ep,cot='One short sentence. '*200),'both_cot')
    assert m['cot_truncated'] and m['cot_tokens']>0 and m['total_tokens']<=c['input']['max_sequence_tokens']
    c['input']['max_sequence_tokens']=base_meta['total_tokens']
    with pytest.raises(ValueError,match='cot_fully_truncated'):proc.endpoint(ep,'both_cot')
    c['input']['max_sequence_tokens']=original_budget


def test_process_boundaries_actual_tokens():
    from teacher_pipeline.common import load_config
    from teacher_pipeline.model import EndpointProcessor
    from teacher_pipeline.process_export import chunk_boundaries
    c=load_config('configs/teacher_pilot.yaml');proc=EndpointProcessor(c)
    ep=dict(id='process-test',text='Represent this text.',images=[],video_frames=[],cot='<thinking>First evidence. Second relation. Third synthesis.</thinking>')
    inputs,meta=proc.endpoint(ep,'both_cot')
    ends,spans=chunk_boundaries(proc.tokenizer,meta,inputs['input_ids'][0].tolist(),3)
    assert len(ends)==3 and ends==sorted(set(ends))
    assert all(end<meta['pooling_position'] for end in ends)
    short=dict(ep,cot='Cat.')
    inputs,meta=proc.endpoint(short,'both_cot')
    ends,spans=chunk_boundaries(proc.tokenizer,meta,inputs['input_ids'][0].tolist(),3)
    assert 0<len(ends)<3


def test_media_content_grouping(tmp_path):
    from PIL import Image
    from teacher_pipeline.data import UnionFind
    image=Image.new('RGB',(32,32),(12,34,56));image.save(tmp_path/'one.png');image.save(tmp_path/'two.png')
    resolver=MediaResolver({'images/':str(tmp_path)})
    paths=resolver.resolve(['images/one.png','images/two.png','images/blank.jpg'])
    assert len(paths)==2 and resolver.cache[paths[0]]==resolver.cache[paths[1]]
    uf=UnionFind();uf.union('query-a','media-a');uf.union('query-b','media-a')
    assert uf.find('query-a')==uf.find('query-b')
    assert uf.find('answer-dog')!=uf.find('query-a')
