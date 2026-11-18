"""Head masking for HuggingFace T5.

Each T5 attention block concatenates its per-head outputs and then applies the
output projection `o`. We register a forward *pre*-hook on `o` that multiplies
each head's slice by a mask value. Mask = 1 keeps a head, 0 removes it.

Because the mask is a differentiable tensor, d(loss)/d(mask) gives the
head-importance score of Michel et al. (2019), "Are Sixteen Heads Really
Better Than One?".

This works across transformers versions because it does not depend on the
(deprecated) `head_mask` arguments of `model.forward`.
"""
import torch

GROUPS = ("enc_self", "dec_self", "dec_cross")


def attention_modules(model):
    """Return {group_name: [T5Attention per layer]}."""
    return {
        "enc_self": [blk.layer[0].SelfAttention for blk in model.encoder.block],
        "dec_self": [blk.layer[0].SelfAttention for blk in model.decoder.block],
        "dec_cross": [blk.layer[1].EncDecAttention for blk in model.decoder.block],
    }


class HeadMasker:
    def __init__(self, model):
        self.model = model
        self.n_heads = model.config.num_heads
        self.d_kv = model.config.d_kv
        self.modules = attention_modules(model)
        device = next(model.parameters()).device
        self.mask = {
            g: torch.ones(len(mods), self.n_heads, device=device)
            for g, mods in self.modules.items()
        }
        self._handles = []
        for g, mods in self.modules.items():
            for layer, attn in enumerate(mods):
                self._handles.append(
                    attn.o.register_forward_pre_hook(self._make_hook(g, layer))
                )

    def _make_hook(self, group, layer):
        def hook(module, args):
            x = args[0]
            m = self.mask[group][layer].to(device=x.device, dtype=x.dtype)
            shape = x.shape
            x = x.view(*shape[:-1], self.n_heads, self.d_kv) * m.view(-1, 1)
            return (x.view(shape),) + tuple(args[1:])
        return hook

    # ---- bookkeeping -------------------------------------------------------
    @property
    def total_heads(self):
        return sum(m.numel() for m in self.mask.values())

    @property
    def n_pruned(self):
        return int(sum((m == 0).sum().item() for m in self.mask.values()))

    @property
    def sparsity(self):
        return self.n_pruned / self.total_heads

    def reset(self):
        for m in self.mask.values():
            m.fill_(1.0)

    def set_requires_grad(self, flag):
        for g in self.mask:
            self.mask[g] = self.mask[g].detach().requires_grad_(flag)

    def to_lists(self):
        return {g: m.detach().cpu().int().tolist() for g, m in self.mask.items()}

    def load_lists(self, lists):
        for g, rows in lists.items():
            self.mask[g] = torch.tensor(rows, dtype=torch.float32,
                                        device=self.mask[g].device)

    def pruned_heads(self):
        """{group: {layer: [head indices with mask 0]}}"""
        out = {}
        for g, m in self.mask.items():
            out[g] = {l: (m[l] == 0).nonzero().flatten().tolist()
                      for l in range(m.shape[0])}
        return out

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
