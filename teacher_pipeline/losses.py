"""Symmetric multi-positive contrastive objective, with explicit valid masks."""
import torch


def positive_mask(rows, candidates, device=None):
    return torch.tensor([[c['id'] in r['positive_candidate_ids'] for c in candidates] for r in rows],device=device,dtype=torch.bool)


def contrastive_loss(q, d, positive, temperature=0.02, valid=None):
    scores=q.float() @ d.float().T / temperature
    positive=positive.to(scores.device).bool()
    valid=torch.ones_like(positive) if valid is None else valid.to(scores.device).bool()
    if positive.shape != scores.shape or valid.shape != scores.shape:
        raise ValueError('mask shape mismatch')
    if (positive & ~valid).any(): raise ValueError('positive outside valid mask')
    def direction(s,p,v):
        keep=p.any(dim=1) & v.any(dim=1)
        if not keep.any(): return s.sum()*0, 0
        s,p,v=s[keep],p[keep],v[keep]
        values=torch.logsumexp(s.masked_fill(~v,-torch.inf),1)-torch.logsumexp(s.masked_fill(~p,-torch.inf),1)
        return values.mean(), int(keep.sum())
    a,nq=direction(scores,positive,valid);b,nd=direction(scores.T,positive.T,valid.T)
    if not nq or not nd: raise ValueError('contrastive batch has no positives')
    if not (valid & ~positive).any(): raise ValueError('contrastive batch has no negatives')
    return (a+b)/2,dict(queries=len(q),candidates=len(d),valid_queries=nq,valid_candidates=nd,
                        skipped_queries=len(q)-nq,skipped_candidates=len(d)-nd,
                        negatives=int((valid & ~positive).sum()))


def gradcache_backward(encode, query_chunks, candidate_chunks, positive, temperature):
    """Exact first-order gradient replay, including dropout RNG, no parameter update between passes."""
    chunks=query_chunks+candidate_chunks
    reps=[]; states=[]
    for chunk in chunks:
        states.append((torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        with torch.no_grad(): reps.append(encode(chunk).detach())
    leaves=[r.requires_grad_() for r in reps]
    split=len(query_chunks)
    loss,stats=contrastive_loss(torch.cat(leaves[:split]),torch.cat(leaves[split:]),positive,temperature)
    loss.backward()
    # fork_rng restores the post-first-pass stream, matching an ordinary forward.
    for chunk,leaf,(cpu,cuda) in zip(chunks,leaves,states):
        with torch.random.fork_rng():
            torch.set_rng_state(cpu)
            if cuda: torch.cuda.set_rng_state_all(cuda)
            current=encode(chunk)
            (current * leaf.grad).sum().backward()
    return loss.detach(),stats
