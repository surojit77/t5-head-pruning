"""Measure real efficiency gains from pruning.

Compares, for the same inputs:
  dense            original model
  masked           heads zeroed by the mask (same size/speed as dense; shows that
                   masking alone gives no speed-up)
  pruned           heads physically removed (real size/FLOP/latency savings)

    python run_benchmark.py --task translation --method gradient --target 0.4
    python run_benchmark.py --task translation --method gradient --all_levels
"""
import argparse
import copy
import gc
import json
import os

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from t5prune.benchmark import (encoder_latency, forward_flops,
                               generation_latency, param_counts)
from t5prune.data import TASKS, generate, load_split, score
from t5prune.heads import HeadMasker
from t5prune.physical import physically_prune


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--method", default="gradient")
    p.add_argument("--target", type=float, default=0.4)
    p.add_argument("--all_levels", action="store_true",
                   help="benchmark every sparsity level of --method")
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 16])
    p.add_argument("--src_len", type=int, default=128)
    p.add_argument("--new_tokens", type=int, default=64)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--n_verify", type=int, default=100,
                   help="examples used to confirm pruned == masked quality")
    p.add_argument("--results", default="results")
    return p.parse_args()


def make_inputs(tok, cfg, sources, bs, src_len, device):
    texts = [cfg.prefix + sources[i % len(sources)] for i in range(bs)]
    enc = tok(texts, max_length=src_len, truncation=True,
              padding="max_length", return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def free(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def measure(model, tok, cfg, sources, a, device):
    """Only `model` is on the GPU while this runs, so peak memory is its own."""
    out = param_counts(model)
    out["weights_mb"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    enc1 = make_inputs(tok, cfg, sources, 1, a.src_len, device)
    out["gflops_forward"] = forward_flops(model, enc1, a.new_tokens) / 1e9
    for bs in a.batch_sizes:
        enc = make_inputs(tok, cfg, sources, bs, a.src_len, device)
        r = generation_latency(model, enc, a.new_tokens)
        r.update(encoder_latency(model, enc))
        out[f"bs{bs}"] = r
    return out


def main():
    a = parse_args()
    sweep = json.load(open(os.path.join(a.results, f"sweep_{a.task}.json")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (a.fp16 and device == "cuda") else torch.float32
    cfg = TASKS[a.task]
    tok = AutoTokenizer.from_pretrained(sweep["model"])
    # the original model stays on the CPU; each variant is copied to the GPU,
    # measured alone, then deleted, so memory numbers are not mixed up
    base = T5ForConditionalGeneration.from_pretrained(sweep["model"], dtype=dtype).eval()
    sources, refs = load_split(a.task, "test", max(a.n_verify, 64))

    runs = [r for r in sweep["runs"] if r["method"] == a.method and r["seed"] == 0]
    if not a.all_levels:
        runs = [r for r in runs if abs(r["target"] - a.target) < 1e-6]
    if not runs:
        raise SystemExit(f"no run for method={a.method} target={a.target} in sweep file")

    report = {"model": sweep["model"], "task": a.task, "device": device,
              "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
              "dtype": str(dtype), "src_len": a.src_len, "new_tokens": a.new_tokens}
    print("measuring dense ...")
    dense = copy.deepcopy(base).to(device)
    report["dense"] = measure(dense, tok, cfg, sources, a, device)
    del dense
    free(device)

    # masked model: same weights, heads zeroed -> expect ~no speed-up
    first = runs[-1]
    masked = copy.deepcopy(base).to(device)
    masker = HeadMasker(masked)
    masker.load_lists(first["mask"])
    print("measuring masked ...")
    report["masked"] = {"target": first["target"], **measure(masked, tok, cfg, sources, a, device)}
    del masked, masker
    free(device)

    report["pruned"] = []
    for run in runs:
        pruned = physically_prune(copy.deepcopy(base), run["mask"]).to(device)
        print(f"measuring pruned @ {run['sparsity']:.2f} ...")
        m = measure(pruned, tok, cfg, sources, a, device)
        # confirm physical pruning reproduces the masked model's quality
        preds = generate(pruned, tok, cfg, sources[:a.n_verify], 16, device)
        m["verify_metrics"] = score(a.task, preds, refs[:a.n_verify])
        report["pruned"].append({"target": run["target"], "sparsity": run["sparsity"],
                                 "sweep_metrics": run["metrics"], **m})
        del pruned
        free(device)

    d = report["dense"]
    for p in report["pruned"]:
        p["param_reduction_pct"] = 100 * (1 - p["total_params"] / d["total_params"])
        p["attn_param_reduction_pct"] = 100 * (1 - p["attention_params"] / d["attention_params"])
        p["flop_reduction_pct"] = 100 * (1 - p["gflops_forward"] / d["gflops_forward"])
        for bs in a.batch_sizes:
            p[f"bs{bs}"]["speedup"] = d[f"bs{bs}"]["latency_ms"] / p[f"bs{bs}"]["latency_ms"]

    path = os.path.join(a.results, f"benchmark_{a.task}_{a.method}.json")
    json.dump(report, open(path, "w"), indent=1)

    print(f"\n{'model':>14} {'params(M)':>10} {'GFLOPs':>8} " +
          " ".join(f"{'bs'+str(b)+' ms':>10} {'speedup':>7}" for b in a.batch_sizes))
    def row(name, r, sp=None):
        cells = [f"{name:>14}", f"{r['total_params']/1e6:10.1f}", f"{r['gflops_forward']:8.2f}"]
        for b in a.batch_sizes:
            s = d[f"bs{b}"]["latency_ms"] / r[f"bs{b}"]["latency_ms"]
            cells += [f"{r[f'bs{b}']['latency_ms']:10.1f}", f"{s:7.2f}x"]
        print(" ".join(cells))
    row("dense", d)
    row(f"masked@{report['masked']['target']:.0%}", report["masked"])
    for p in report["pruned"]:
        row(f"pruned@{p['sparsity']:.0%}", p)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
