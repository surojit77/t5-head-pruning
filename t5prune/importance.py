"""Head-importance scores and the pruning step.

gradient : Michel et al. (2019).  I_h = sum over batches |dL/d mask_h|,
           then L2-normalised within each layer.
magnitude: norm of the head's slice of the Q, K, V and O weights.
random   : uniform random scores (a sanity baseline).
"""
import torch


def _normalize_per_layer(scores):
    return {g: s / (s.norm(dim=-1, keepdim=True) + 1e-20) for g, s in scores.items()}


def gradient_importance(model, masker, batches, normalize=True):
    model.eval()                      # no dropout while scoring
    masker.set_requires_grad(True)
    groups = list(masker.mask)
    scores = {g: torch.zeros_like(masker.mask[g]) for g in groups}
    for batch in batches:
        loss = model(**batch).loss
        grads = torch.autograd.grad(loss, [masker.mask[g] for g in groups])
        for g, gr in zip(groups, grads):
            scores[g] += gr.abs().detach()
    masker.set_requires_grad(False)
    return _normalize_per_layer(scores) if normalize else scores


@torch.no_grad()
def magnitude_importance(masker, normalize=True):
    d, H = masker.d_kv, masker.n_heads
    scores = {}
    for g, mods in masker.modules.items():
        s = torch.zeros(len(mods), H, device=masker.mask[g].device)
        for l, attn in enumerate(mods):
            for h in range(H):
                rows = slice(h * d, (h + 1) * d)
                sq = sum(lin.weight[rows].float().pow(2).sum()
                         for lin in (attn.q, attn.k, attn.v))
                sq = sq + attn.o.weight[:, rows].float().pow(2).sum()
                s[l, h] = sq.sqrt()
        scores[g] = s
    return _normalize_per_layer(scores) if normalize else scores


def random_importance(masker, seed):
    gen = torch.Generator().manual_seed(seed)
    return {g: torch.rand(m.shape, generator=gen).to(m.device)
            for g, m in masker.mask.items()}


@torch.no_grad()
def prune_to(masker, scores, n_target, min_heads_per_layer=1):
    """Mask the lowest-scoring *remaining* heads until n_target heads are pruned
    in total. Ranking is global across all layers and attention types. Each
    attention layer keeps at least `min_heads_per_layer` heads (needed so the
    pruned model can be physically shrunk later)."""
    candidates = []
    for g, m in masker.mask.items():
        for l in range(m.shape[0]):
            for h in range(m.shape[1]):
                if m[l, h] == 1:
                    candidates.append((scores[g][l, h].item(), g, l, h))
    candidates.sort()
    for _, g, l, h in candidates:
        if masker.n_pruned >= n_target:
            break
        if masker.mask[g][l].sum() > min_heads_per_layer:
            masker.mask[g][l, h] = 0.0
