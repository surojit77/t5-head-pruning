"""Physically remove pruned heads so the model really gets smaller and faster.

Masking a head (heads.py) leaves the model the same size and speed: the
computation still happens and is then multiplied by zero. To measure real
gains we slice the pruned heads' rows out of Q, K, V and their columns out of O.

T5 detail: the relative position bias is computed once (in layer 0) with one
column per head and is then shared by every layer. After pruning, layers keep
different numbers of heads, so each pruned layer receives the full bias, takes
only its kept heads, and passes the full bias on unchanged.
(transformers v5 removed the built-in `prune_heads`, so we do this ourselves.)
"""
import types

import torch
from torch import nn

from .heads import attention_modules


def _slice_linear(lin, idx, dim):
    W = lin.weight.data.index_select(dim, idx).clone()
    new = nn.Linear(W.shape[1], W.shape[0], bias=False,
                    device=W.device, dtype=W.dtype)
    new.weight.data.copy_(W)
    return new


def _prune_attention(attn, keep_heads):
    H, d = attn.n_heads, attn.key_value_proj_dim
    keep = torch.tensor(sorted(keep_heads), dtype=torch.long,
                        device=attn.q.weight.device)
    rows = (keep[:, None] * d + torch.arange(d, device=keep.device)).flatten()
    attn.q = _slice_linear(attn.q, rows, 0)
    attn.k = _slice_linear(attn.k, rows, 0)
    attn.v = _slice_linear(attn.v, rows, 0)
    attn.o = _slice_linear(attn.o, rows, 1)
    attn.full_heads = H
    attn.kept_heads = keep
    attn.n_heads = len(keep)
    attn.inner_dim = attn.n_heads * d

    # the shared position-bias table stays full-size; slice it per layer
    orig_compute_bias = attn.compute_bias

    def compute_bias(self, *a, **kw):
        full = orig_compute_bias(*a, **kw)
        self._full_bias = full
        return full[:, self.kept_heads]

    orig_forward = attn.forward

    def forward(self, hidden_states, mask=None, key_value_states=None,
                position_bias=None, past_key_values=None, **kw):
        self._full_bias = None
        pb = position_bias
        if pb is not None and pb.shape[1] == self.full_heads:
            pb = pb[:, self.kept_heads]
        out, ret_bias, w = orig_forward(hidden_states, mask, key_value_states,
                                        pb, past_key_values, **kw)
        if position_bias is not None:
            ret_bias = position_bias                    # pass on unchanged
        elif self._full_bias is not None:
            ret_bias = self._full_bias                  # full relative bias
        else:
            ret_bias = ret_bias[:, :1]                  # all-zero bias, broadcastable
        return out, ret_bias, w

    if attn.has_relative_attention_bias:
        attn.compute_bias = types.MethodType(compute_bias, attn)
    attn.forward = types.MethodType(forward, attn)


@torch.no_grad()
def physically_prune(model, mask_lists):
    """mask_lists: {group: [[0/1 per head] per layer]} as saved by HeadMasker.to_lists()."""
    for group, mods in attention_modules(model).items():
        for layer, attn in enumerate(mods):
            row = mask_lists[group][layer]
            keep = [h for h, m in enumerate(row) if m]
            if not keep:
                raise ValueError(f"{group} layer {layer}: every head pruned")
            if len(keep) < len(row):
                _prune_attention(attn, keep)
    return model
