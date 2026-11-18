"""Prune T5 attention heads at increasing sparsity and evaluate quality.

Example (Colab T4, ~1-2 h for both tasks):
    python run_sweep.py --task translation
    python run_sweep.py --task summarization --n_eval 300

Quick smoke run (~5 min):
    python run_sweep.py --task translation --quick

Methods
  gradient          Michel et al. importance, re-computed after every pruning step (iterative)
  gradient_oneshot  same score, computed once on the dense model
  magnitude         norm of each head's Q/K/V/O weights
  random            random order, averaged over --seeds seeds
Importance is computed on the validation split; quality is measured on the test split.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from t5prune.data import (PRIMARY_METRIC, TASKS, generate, load_split,
                          make_loss_batches, score)
from t5prune.heads import HeadMasker
from t5prune.importance import (gradient_importance, magnitude_importance,
                                prune_to, random_importance)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google-t5/t5-small")
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--methods", nargs="+",
                   default=["gradient", "gradient_oneshot", "magnitude", "random"])
    p.add_argument("--levels", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--seeds", type=int, default=3, help="seeds for the random baseline")
    p.add_argument("--n_calib", type=int, default=256)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_beams", type=int, default=None, help="override beam size")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.quick:
        a.n_calib, a.n_eval, a.seeds, a.num_beams = 64, 100, 1, 1
        a.levels = [0.0, 0.2, 0.4, 0.6]
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"sweep_{a.task}.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TASKS[a.task]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = T5ForConditionalGeneration.from_pretrained(a.model).to(device).eval()
    model.requires_grad_(False)          # we only need gradients w.r.t. the head mask
    masker = HeadMasker(model)

    calib_src, calib_tgt = load_split(a.task, "validation", a.n_calib)
    eval_src, eval_ref = load_split(a.task, "test", a.n_eval)
    calib = make_loss_batches(tok, cfg, calib_src, calib_tgt, 16, device)

    # resume support (Colab disconnects)
    results = json.load(open(out_path)) if os.path.exists(out_path) else {
        "model": a.model, "task": a.task, "total_heads": masker.total_heads,
        "n_eval": a.n_eval, "runs": []}
    done = {(r["method"], r["seed"], r["target"]) for r in results["runs"]}

    def evaluate():
        t = time.time()
        preds = generate(model, tok, cfg, eval_src, a.batch_size, device, a.num_beams)
        m = score(a.task, preds, eval_ref)
        m["eval_seconds"] = round(time.time() - t, 1)
        return m, preds[:3]

    # dense baseline (shared by every method)
    if ("dense", 0, 0.0) not in done:
        masker.reset()
        m, ex = evaluate()
        results["runs"].append({"method": "dense", "seed": 0, "target": 0.0,
                                "sparsity": 0.0, "metrics": m, "examples": ex,
                                "mask": masker.to_lists()})
        json.dump(results, open(out_path, "w"), indent=1)
        print(f"dense: {m}")

    for method in a.methods:
        seeds = range(a.seeds) if method == "random" else [0]
        for seed in seeds:
            masker.reset()
            scores = None
            for target in sorted(l for l in a.levels if l > 0):
                n_target = round(target * masker.total_heads)
                if method == "gradient":
                    scores = gradient_importance(model, masker, calib)
                elif scores is None:
                    scores = {"gradient_oneshot": lambda: gradient_importance(model, masker, calib),
                              "magnitude": lambda: magnitude_importance(masker),
                              "random": lambda: random_importance(masker, seed)}[method]()
                prune_to(masker, scores, n_target)
                if (method, seed, target) in done:
                    continue
                m, ex = evaluate()
                results["runs"].append({
                    "method": method, "seed": seed, "target": target,
                    "sparsity": masker.sparsity, "metrics": m, "examples": ex,
                    "mask": masker.to_lists(),
                    "importance": {g: s.cpu().tolist() for g, s in scores.items()},
                })
                json.dump(results, open(out_path, "w"), indent=1)
                key = PRIMARY_METRIC[a.task]
                print(f"{method:17s} seed={seed} sparsity={masker.sparsity:.2f} "
                      f"{key}={m[key]:.2f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
