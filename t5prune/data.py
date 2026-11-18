"""Datasets, batching and metrics.

Pre-trained T5 checkpoints were trained on a multi-task mixture that already
includes WMT16 En->Ro and CNN/DailyMail, so the original prompts work without
fine-tuning.
"""
from dataclasses import dataclass

import sacrebleu
import torch
from rouge_score import rouge_scorer


@dataclass
class TaskConfig:
    name: str
    prefix: str
    max_source_len: int
    max_target_len: int
    max_new_tokens: int
    gen_kwargs: dict


TASKS = {
    "translation": TaskConfig(
        "translation", "translate English to Romanian: ", 256, 128, 128,
        dict(num_beams=4, early_stopping=True)),
    "summarization": TaskConfig(
        "summarization", "summarize: ", 512, 128, 128,
        dict(num_beams=4, early_stopping=True, length_penalty=2.0,
             no_repeat_ngram_size=3, min_new_tokens=30)),
}


def load_split(task, split, n, seed=0):
    """Return (sources, references). Calibration uses 'validation', eval uses 'test'."""
    from datasets import load_dataset
    if task == "translation":
        ds = load_dataset("wmt/wmt16", "ro-en", split=split)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return ([ex["translation"]["en"] for ex in ds],
                [ex["translation"]["ro"] for ex in ds])
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return list(ds["article"]), list(ds["highlights"])


def make_loss_batches(tok, cfg, sources, targets, batch_size, device):
    """Teacher-forced batches (with labels) for computing gradient importance."""
    batches = []
    for i in range(0, len(sources), batch_size):
        enc = tok([cfg.prefix + s for s in sources[i:i + batch_size]],
                  max_length=cfg.max_source_len, truncation=True,
                  padding=True, return_tensors="pt")
        lab = tok(text_target=targets[i:i + batch_size],
                  max_length=cfg.max_target_len, truncation=True,
                  padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        batches.append({"input_ids": enc.input_ids.to(device),
                        "attention_mask": enc.attention_mask.to(device),
                        "labels": lab.to(device)})
    return batches


@torch.no_grad()
def generate(model, tok, cfg, sources, batch_size, device, num_beams=None):
    model.eval()
    kwargs = dict(cfg.gen_kwargs)
    if num_beams is not None:
        kwargs["num_beams"] = num_beams
        if num_beams == 1:
            kwargs.pop("early_stopping", None)
            kwargs.pop("length_penalty", None)
    preds = []
    for i in range(0, len(sources), batch_size):
        enc = tok([cfg.prefix + s for s in sources[i:i + batch_size]],
                  max_length=cfg.max_source_len, truncation=True,
                  padding=True, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, **kwargs)
        preds += tok.batch_decode(out, skip_special_tokens=True)
    return preds


_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score(task, preds, refs):
    if task == "translation":
        return {"bleu": sacrebleu.corpus_bleu(preds, [refs]).score,
                "chrf": sacrebleu.corpus_chrf(preds, [refs]).score}
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(preds, refs):
        s = _ROUGE.score(r, p)
        for k in totals:
            totals[k] += s[k].fmeasure
    return {k: 100 * v / len(preds) for k, v in totals.items()}


PRIMARY_METRIC = {"translation": "bleu", "summarization": "rougeL"}
