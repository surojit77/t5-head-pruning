"""Size, FLOPs, latency, throughput and memory measurements."""
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from .heads import attention_modules


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    attn = 0
    for mods in attention_modules(model).values():
        for a in mods:
            attn += sum(lin.weight.numel() for lin in (a.q, a.k, a.v, a.o))
    return {"total_params": total, "attention_params": attn}


@torch.no_grad()
def forward_flops(model, enc, decoder_len=64):
    """FLOPs of one teacher-forced forward pass (encoder + decoder)."""
    dec = torch.full((enc["input_ids"].shape[0], decoder_len),
                     getattr(model.config, "decoder_start_token_id", None) or 0,
                     device=enc["input_ids"].device)
    with FlopCounterMode(display=False) as fc:
        model(**enc, decoder_input_ids=dec)
    return fc.get_total_flops()


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generation_latency(model, enc, new_tokens=64, warmup=3, repeats=10):
    """Greedy generation with a FIXED output length so every model does the
    same amount of work (a pruned model might otherwise stop earlier/later)."""
    device = enc["input_ids"].device.type
    kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens,
              num_beams=1, do_sample=False)
    for _ in range(warmup):
        model.generate(**enc, **kw)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        _sync(device)
        t = time.perf_counter()
        model.generate(**enc, **kw)
        _sync(device)
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    out = {"latency_ms": 1000 * med,
           "latency_ms_std": 1000 * statistics.pstdev(times),
           "tokens_per_s": enc["input_ids"].shape[0] * new_tokens / med}
    if device == "cuda":
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out


@torch.no_grad()
def encoder_latency(model, enc, warmup=3, repeats=20):
    device = enc["input_ids"].device.type
    for _ in range(warmup):
        model.encoder(**enc)
    times = []
    for _ in range(repeats):
        _sync(device)
        t = time.perf_counter()
        model.encoder(**enc)
        _sync(device)
        times.append(time.perf_counter() - t)
    return {"encoder_ms": 1000 * statistics.median(times)}
