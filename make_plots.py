"""Make figures + a markdown summary table from results/*.json.

    python make_plots.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from t5prune.data import PRIMARY_METRIC
from t5prune.plots import finetune_bars, pruned_heads_heatmap, tradeoff_curve


def summary_table(sweep_path):
    res = json.load(open(sweep_path))
    key = PRIMARY_METRIC[res["task"]]
    dense = next(r for r in res["runs"] if r["method"] == "dense")["metrics"][key]
    by = defaultdict(lambda: defaultdict(list))
    for r in res["runs"]:
        if r["method"] != "dense":
            by[r["method"]][r["target"]].append(r["metrics"][key])
    levels = sorted({t for m in by.values() for t in m})
    lines = [f"### {res['task']} ({key}, dense = {dense:.2f}, n = {res['n_eval']})", "",
             "| method | " + " | ".join(f"{int(100*l)}%" for l in levels) + " |",
             "|---|" + "---|" * len(levels)]
    for method, pts in by.items():
        cells = []
        for l in levels:
            v = pts.get(l)
            if not v:
                cells.append("–")
            else:
                cells.append(f"{np.mean(v):.2f} ({100*np.mean(v)/dense:.0f}%)")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(results="results"):
    md = []
    for path in sorted(glob.glob(os.path.join(results, "sweep_*.json"))):
        task = json.load(open(path))["task"]
        tradeoff_curve(path, os.path.join(results, f"tradeoff_{task}.png"))
        try:
            pruned_heads_heatmap(path, os.path.join(results, f"heads_{task}.png"))
        except StopIteration:
            pass
        md.append(summary_table(path))
    for path in sorted(glob.glob(os.path.join(results, "benchmark_*.json"))):
        b = json.load(open(path))
        bs = [k for k in b["dense"] if k.startswith("bs")]
        start = len(md)
        md.append(f"### Efficiency — {b['task']} on {b['gpu']} ({b['dtype']})\n")
        md.append(f"| model | params (M) | attn params (M) | GFLOPs | peak GPU MB ({bs[-1]}) | " +
                  " | ".join(f"{k} latency (ms) / speed-up" for k in bs) + " |")
        md.append("|---|---|---|---|---|" + "---|" * len(bs))
        def row(name, r):
            cells = [name, f"{r['total_params']/1e6:.1f}", f"{r['attention_params']/1e6:.1f}",
                     f"{r['gflops_forward']:.2f}"]
            last = r[bs[-1]]
            cells.append(f"{last['peak_mem_mb']:.0f}" if "peak_mem_mb" in last else "–")
            for k in bs:
                sp = b["dense"][k]["latency_ms"] / r[k]["latency_ms"]
                cells.append(f"{r[k]['latency_ms']:.1f} / {sp:.2f}x")
            md.append("| " + " | ".join(cells) + " |")
        row("dense", b["dense"])
        row(f"masked {b['masked']['target']:.0%} (no removal)", b["masked"])
        for p in b["pruned"]:
            row(f"pruned {p['sparsity']:.0%}", p)
        md[start:] = ["\n".join(md[start:])]
    for path in sorted(glob.glob(os.path.join(results, "finetune_*.json"))):
        f = json.load(open(path))
        key = PRIMARY_METRIC[f["task"]]
        finetune_bars(path, os.path.join(results, f"finetune_{f['task']}.png"))
        runs = sorted(f["runs"], key=lambda r: r["target"])
        dense = next((r for r in runs if r["target"] == 0.0), None)
        rows = [f"### Recovery fine-tuning — {f['task']} ({key}; {f['n_train']} train examples, "
                f"{f['epochs']} epoch, lr {f['lr']})\n",
                "| model | after pruning | after fine-tuning | % of dense | % of dense + same FT |",
                "|---|---|---|---|---|"]
        for r in runs:
            b, a = r["before"][key], r["after"][key]
            nm = "dense" if r["target"] == 0 else f"pruned {r['sparsity']:.0%}"
            pd = f"{100 * a / dense['before'][key]:.0f}%" if dense else "–"
            pf = f"{100 * a / dense['after'][key]:.0f}%" if dense else "–"
            rows.append(f"| {nm} | {b:.2f} | {a:.2f} | {pd} | {pf} |")
        md.append("\n".join(rows))
    out = os.path.join(results, "SUMMARY.md")
    open(out, "w").write("\n\n".join(md) + "\n")
    print(open(out).read())


if __name__ == "__main__":
    main()
