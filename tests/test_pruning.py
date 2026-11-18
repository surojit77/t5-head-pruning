"""Correctness checks on a tiny randomly-initialised T5 (no downloads).

    python -m pytest tests -q      (or: python tests/test_pruning.py)

1. Hooks with an all-ones mask do not change the model's output.
2. Masking heads == physically removing them (same logits, same generations),
   including layer 0, which owns the shared relative-position bias.
3. Gradient / magnitude / random scoring and prune_to behave as expected.
4. Physical pruning reduces parameters and FLOPs.
"""
import copy
import os
import sys

import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t5prune.benchmark import forward_flops, param_counts  # noqa: E402
from t5prune.heads import HeadMasker  # noqa: E402
from t5prune.importance import (gradient_importance, magnitude_importance,  # noqa: E402
                                prune_to, random_importance)
from t5prune.physical import physically_prune  # noqa: E402


def tiny_model(attn_impl="eager"):
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=100, d_model=32, d_kv=8, num_heads=4, d_ff=64,
                   num_layers=3, num_decoder_layers=3, dropout_rate=0.0,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    cfg._attn_implementation = attn_impl
    return T5ForConditionalGeneration(cfg).eval()


def batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(2, 100, (2, 12), generator=g)
    am = torch.ones_like(ids)
    am[1, 9:] = 0
    labels = torch.randint(2, 100, (2, 7), generator=g)
    return {"input_ids": ids, "attention_mask": am, "labels": labels}


MASK = {"enc_self": [[0, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]],
        "dec_self": [[1, 0, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]],
        "dec_cross": [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]]}


def test_identity_mask():
    m = tiny_model()
    b = batch()
    ref = m(**b).logits
    masker = HeadMasker(m)
    assert torch.allclose(ref, m(**b).logits, atol=1e-6)
    masker.remove()


def _check_mask_equals_physical(attn_impl):
    masked = tiny_model(attn_impl)
    physical = copy.deepcopy(masked)
    masker = HeadMasker(masked)
    masker.load_lists(MASK)
    b = batch()
    with torch.no_grad():
        dense_logits = physical(**b).logits
        a = masked(**b).logits
        assert not torch.allclose(a, dense_logits, atol=1e-3)  # mask really bites
        physically_prune(physical, MASK)
        p = physical(**b).logits
    assert torch.allclose(a, p, atol=1e-5), (a - p).abs().max()
    enc = {k: b[k] for k in ("input_ids", "attention_mask")}
    ga = masked.generate(**enc, max_new_tokens=8, num_beams=1, do_sample=False)
    gp = physical.generate(**enc, max_new_tokens=8, num_beams=1, do_sample=False)
    assert torch.equal(ga, gp)
    gb = masked.generate(**enc, max_new_tokens=8, num_beams=3)
    gpb = physical.generate(**enc, max_new_tokens=8, num_beams=3)
    assert torch.equal(gb, gpb)


def test_mask_equals_physical_eager():
    _check_mask_equals_physical("eager")


def test_mask_equals_physical_sdpa():
    _check_mask_equals_physical("sdpa")


def test_importance_and_prune():
    m = tiny_model()
    m.requires_grad_(False)
    masker = HeadMasker(m)
    b = batch()
    for scores in (gradient_importance(m, masker, [b, b]),
                   magnitude_importance(masker), random_importance(masker, 0)):
        assert set(scores) == set(masker.mask)
        assert all(s.shape == masker.mask[g].shape for g, s in scores.items())
    scores = gradient_importance(m, masker, [b])
    assert all((s > 0).all() for s in scores.values())
    prune_to(masker, scores, 10)
    assert masker.n_pruned == 10
    # lowest-scored heads go first
    masker.reset()
    crafted = {g: torch.arange(s.numel(), dtype=torch.float).view(s.shape) + 100 * i
               for i, (g, s) in enumerate(masker.mask.items())}
    prune_to(masker, crafted, 2)
    assert masker.mask["enc_self"][0].tolist() == [0, 0, 1, 1]
    # every layer keeps at least one head even at extreme sparsity
    prune_to(masker, scores, masker.total_heads)
    assert all((m.sum(-1) >= 1).all() for m in masker.mask.values())


def test_size_and_flops_shrink():
    dense = tiny_model()
    pruned = physically_prune(copy.deepcopy(dense), MASK)
    b = batch()
    enc = {k: b[k] for k in ("input_ids", "attention_mask")}
    assert param_counts(pruned)["attention_params"] < param_counts(dense)["attention_params"]
    assert forward_flops(pruned, enc, 8) < forward_flops(dense, enc, 8)


def test_pruned_model_trains():
    """A physically pruned model can be fine-tuned and pruned heads stay gone."""
    m = physically_prune(tiny_model(), MASK)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    b = batch()
    first = None
    for _ in range(30):
        loss = m(**b).loss
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < first
    m.eval()
    assert m.encoder.block[0].layer[0].SelfAttention.q.weight.shape[0] == 2 * 8
    enc = {k: b[k] for k in ("input_ids", "attention_mask")}
    m.generate(**enc, max_new_tokens=5, num_beams=2)


def test_masker_follows_device_move():
    m = tiny_model()
    masker = HeadMasker(m)
    masker.load_lists(MASK)
    m.to(torch.float64)          # stands in for a device move: hook must adapt
    b = batch()
    m(**b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
