"""Recovery fine-tuning: how much quality comes back if the pruned model is
briefly trained after pruning?

For each sparsity level, the heads chosen in the sweep are physically removed and
the smaller model is fine-tuned on the task's TRAIN split, then evaluated on the
same test examples as the sweep.

Control: the dense (unpruned) model gets exactly the same fine-tuning. Otherwise
a gain could come from the extra task training, not from "recovering" what the
pruning removed. Report recovery against dense+FT, not only against dense.

    python run_finetune.py --task translation
    python run_finetune.py --task summarization
"""
import argparse
import json
import math
import os
import random
import time

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from t5prune.data import (PRIMARY_METRIC, TASKS, generate, load_split,
                          make_loss_batches, score)
from t5prune.physical import physically_prune

DEFAULT_BS = {"translation": 32, "summarization": 8}
DEFAULT_N = {"translation": 20000, "summarization": 8000}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--method", default="gradient")
    p.add_argument("--levels", nargs="+", type=float, default=[0.4, 0.5, 0.6])
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results")
    p.add_argument("--no_dense_control", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def train(model, tok, cfg, src, tgt, epochs, lr, bs, device, seed, log_every=100):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.train()
    model.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(src) / bs)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    losses, step, t0 = [], 0, time.time()
    for _ in range(epochs):
        order = list(range(len(src)))
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            batch = make_loss_batches(tok, cfg, [src[j] for j in idx],
                                      [tgt[j] for j in idx], bs, device)[0]
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % log_every == 0 or step == steps:
                print(f"  step {step}/{steps} loss {sum(losses[-log_every:]) / len(losses[-log_every:]):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return {"steps": steps, "train_seconds": round(time.time() - t0, 1),
            "loss_first100": sum(losses[:100]) / len(losses[:100]),
            "loss_last100": sum(losses[-100:]) / len(losses[-100:])}


def main():
    a = parse_args()
    a.batch_size = a.batch_size or DEFAULT_BS[a.task]
    a.n_train = a.n_train or DEFAULT_N[a.task]
    sweep = json.load(open(os.path.join(a.results, f"sweep_{a.task}.json")))
    n_eval = sweep["n_eval"]
    if a.quick:
        a.n_train, n_eval, a.levels = 256, 50, a.levels[:1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TASKS[a.task]
    key = PRIMARY_METRIC[a.task]
    tok = AutoTokenizer.from_pretrained(sweep["model"])

    train_src, train_tgt = load_split(a.task, "train", a.n_train, seed=a.seed)
    eval_src, eval_ref = load_split(a.task, "test", n_eval)   # same examples as the sweep

    out_path = os.path.join(a.results, f"finetune_{a.task}.json")
    out = json.load(open(out_path)) if (os.path.exists(out_path) and not a.quick) else {
        "model": sweep["model"], "task": a.task, "method": a.method, "n_eval": n_eval,
        "n_train": a.n_train, "epochs": a.epochs, "lr": a.lr, "batch_size": a.batch_size,
        "runs": []}
    done = {r["target"] for r in out["runs"]}

    variants = [] if a.no_dense_control else [(0.0, None)]
    for lvl in a.levels:
        run = next((r for r in sweep["runs"] if r["method"] == a.method and r["seed"] == 0
                    and abs(r["target"] - lvl) < 1e-6), None)
        if run is None:
            raise SystemExit(f"no {a.method} run at {lvl} in the sweep file")
        variants.append((lvl, run))

    for target, run in variants:
        if target in done:
            continue
        name = "dense" if run is None else f"pruned {target:.0%}"
        print(f"== {name}", flush=True)
        model = T5ForConditionalGeneration.from_pretrained(sweep["model"])
        if run is not None:
            physically_prune(model, run["mask"])
        model.to(device).eval()

        before = score(a.task, generate(model, tok, cfg, eval_src, a.eval_batch_size, device), eval_ref)
        log = train(model, tok, cfg, train_src, train_tgt, a.epochs, a.lr,
                    a.batch_size, device, a.seed)
        preds = generate(model, tok, cfg, eval_src, a.eval_batch_size, device)
        after = score(a.task, preds, eval_ref)
        print(f"   {key}: {before[key]:.2f} -> {after[key]:.2f}", flush=True)
        out["runs"].append({"target": target,
                            "sparsity": 0.0 if run is None else run["sparsity"],
                            "before": before, "after": after, "train": log,
                            "examples": preds[:3]})
        if not a.quick:
            json.dump(out, open(out_path, "w"), indent=1)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    runs = sorted(out["runs"], key=lambda r: r["target"])
    dense = next((r for r in runs if r["target"] == 0.0), None)
    print(f"\n{'model':>12} {key+' before':>12} {key+' after FT':>12}  "
          f"{'% of dense':>10} {'% of dense+FT':>13}")
    for r in runs:
        b, f = r["before"][key], r["after"][key]
        pd = f"{100 * f / dense['before'][key]:9.0f}%" if dense else ""
        pf = f"{100 * f / dense['after'][key]:12.0f}%" if dense else ""
        nm = "dense" if r["target"] == 0 else f"pruned {r['target']:.0%}"
        print(f"{nm:>12} {b:12.2f} {f:12.2f}  {pd:>10} {pf:>13}")
    if not a.quick:
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
