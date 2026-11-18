"""Figures from the sweep JSON files."""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import PRIMARY_METRIC

STYLE = {"gradient": ("#2a6fdb", "o", "Gradient (iterative)"),
         "gradient_oneshot": ("#7aa6ec", "s", "Gradient (one-shot)"),
         "magnitude": ("#e07b39", "^", "Weight magnitude"),
         "random": ("#8a8a8a", "x", "Random (mean ± std)")}


def tradeoff_curve(sweep_path, out_png):
    res = json.load(open(sweep_path))
    key = PRIMARY_METRIC[res["task"]]
    dense = next(r for r in res["runs"] if r["method"] == "dense")["metrics"][key]
    by = defaultdict(lambda: defaultdict(list))
    for r in res["runs"]:
        if r["method"] != "dense":
            by[r["method"]][r["target"]].append(r["metrics"][key])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method, pts in by.items():
        color, marker, label = STYLE.get(method, ("k", ".", method))
        xs = [0.0] + sorted(pts)
        ys = [dense] + [np.mean(pts[x]) for x in xs[1:]]
        ax.plot([100 * x for x in xs], ys, marker=marker, color=color, label=label)
        if method == "random" and any(len(pts[x]) > 1 for x in xs[1:]):
            sd = [0.0] + [np.std(pts[x]) for x in xs[1:]]
            ax.fill_between([100 * x for x in xs], np.array(ys) - sd,
                            np.array(ys) + sd, color=color, alpha=0.15)
    ax.axhline(0.9 * dense, ls="--", lw=1, color="#444")
    ax.text(1, 0.9 * dense, " 90% of dense", va="bottom", fontsize=8, color="#444")
    ax.set_xlabel("Attention heads pruned (%)")
    ax.set_ylabel(key.upper() if key == "bleu" else "ROUGE-L")
    ax.set_title(f"{res['model'].split('/')[-1]} — {res['task']}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def pruned_heads_heatmap(sweep_path, out_png, method="gradient", target=0.4):
    """Which heads the method removed at a given sparsity, per attention type."""
    res = json.load(open(sweep_path))
    run = next(r for r in res["runs"]
               if r["method"] == method and abs(r["target"] - target) < 1e-6)
    groups = list(run["mask"])
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.2))
    titles = {"enc_self": "Encoder self-attn", "dec_self": "Decoder self-attn",
              "dec_cross": "Decoder cross-attn"}
    for ax, g in zip(axes, groups):
        m = np.array(run["mask"][g])
        ax.imshow(m, cmap="Greys_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{titles[g]}\n{int((m == 0).sum())}/{m.size} pruned", fontsize=9)
        ax.set_xticks(range(m.shape[1]))
        ax.set_yticks(range(m.shape[0]))
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
    fig.suptitle(f"{res['task']}: heads kept (white) / pruned (black) at "
                 f"{int(100 * run['sparsity'])}% — {method}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def finetune_bars(ft_path, out_png):
    """Quality before vs after recovery fine-tuning, per sparsity level."""
    res = json.load(open(ft_path))
    key = PRIMARY_METRIC[res["task"]]
    runs = sorted(res["runs"], key=lambda r: r["target"])
    labels = ["dense" if r["target"] == 0 else f"{int(round(100 * r['sparsity']))}% pruned"
              for r in runs]
    before = [r["before"][key] for r in runs]
    after = [r["after"][key] for r in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - 0.2, before, 0.4, label="after pruning", color="#9db8e8")
    ax.bar(x + 0.2, after, 0.4, label="after pruning + fine-tuning", color="#2a6fdb")
    for xi, b, f in zip(x, before, after):
        ax.text(xi - 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, f, f"{f:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BLEU" if key == "bleu" else "ROUGE-L")
    ax.set_title(f"{res['model'].split('/')[-1]} — {res['task']}: recovery fine-tuning "
                 f"({res['n_train']} examples, {res['epochs']} epoch)", fontsize=10)
    ax.set_ylim(0, max(before + after) * 1.18)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
