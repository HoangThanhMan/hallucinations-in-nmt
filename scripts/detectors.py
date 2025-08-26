#!/usr/bin/env python
import json
from collections import Counter
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# Optional metrics
try:
    import sacrebleu
except Exception:
    sacrebleu = None

try:
    from comet import download_model, load_from_checkpoint
except Exception:
    download_model = load_from_checkpoint = None

def top_ngram_count(text, n = 4):
    toks = text.split()
    if len(toks) < n:
        return ("", 0)
    ngrams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    cnt = Counter(ngrams).most_common(1)[0]
    return cnt

def repeated_targets(hyps):
    if not hyps:
        return None
    return dict(Counter(hyps))

def chrf2(hyps, refs):
    if sacrebleu is None:
        return float("nan")
    score = sacrebleu.corpus_chrf(hyps, [refs])
    return score.score

def comet_scores(srcs, hyps, refs=None, model_name="wmt20-comet-da"):
    if load_from_checkpoint is None:
        return [float("nan")]*len(hyps)
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    data = []
    for i in range(len(hyps)):
        item = {"src": srcs[i], "mt": hyps[i]}
        if refs is not None:
            item["ref"] = refs[i]
        data.append(item)
    out = model.predict(data, batch_size=8, gpus=0)
    return out["scores"] if isinstance(out, dict) else out

def average_pairwise_chrf(runs):
    if sacrebleu is None or len(runs) < 2:
        return float("nan")
    import itertools
    pairs = list(itertools.combinations(range(len(runs)), 2))
    sims = []
    for (i, j) in pairs:
        sims.append(chrf2(runs[i], runs[j]))
    return float(np.mean(sims)) if sims else float("nan")

def attn_to_eos(attn, eos_index):
    if attn is None or eos_index < 0:
        return float("nan")
    total = attn.sum()
    return float(attn[:, eos_index].sum() / (total + 1e-12))

def attn_ign_src(attn, threshold):
    if attn is None:
        return float("nan")
    per_src = attn.sum(axis=0)
    ign = (per_src < threshold).mean()
    return float(ign)

def load_attentions(json_path):
    # JSON: list of {"attn": [[...], ...]} per sentence (tgt_len x src_len).
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    outs = []
    for item in data:
        arr = np.array(item["attn"], dtype=float)
        outs.append(arr)
    return outs
