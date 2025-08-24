#!/usr/bin/env python
import argparse, os, sys
import pandas as pd
import numpy as np

# Ensure local import path
sys.path.append(os.path.dirname(__file__))
from detectors import top_ngram_count, repeated_targets_ratio, chrf2, comet_scores

def rule_label(row, tng_thresh=3, seqlog_thresh=-5.0, comet_thresh=0.3, chrf_thresh=45.0):
    # Simple fusion rule to label hallucination.
    conds = []
    if "seq_logprob" in row and pd.notnull(row["seq_logprob"]):
        conds.append(row["seq_logprob"] < seqlog_thresh)
    if "tng_count" in row and pd.notnull(row["tng_count"]):
        conds.append(row["tng_count"] >= tng_thresh)
    if "comet" in row and pd.notnull(row["comet"]):
        conds.append(row["comet"] < comet_thresh)
    if "chrf2" in row and pd.notnull(row["chrf2"]):
        conds.append(row["chrf2"] < chrf_thresh)
    return int(any(conds))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True, help="CSV from 05_parse_generate.py")
    ap.add_argument("--out", required=True, help="Output hallucination dataset CSV")
    ap.add_argument("--annot", default=None, help="Optional: annotated CSV with columns ['id','is_hallu']")
    ap.add_argument("--tng_n", type=int, default=3)
    ap.add_argument("--tng_thresh", type=int, default=3)
    ap.add_argument("--seqlog_thresh", type=float, default=-5.0)
    ap.add_argument("--comet_model", default=None, help="e.g., wmt20-comet-da or wmt20-comet-qe-da-v2")
    ap.add_argument("--comet_thresh", type=float, default=0.3)
    ap.add_argument("--compute_chrf", action="store_true")
    ap.add_argument("--chrf_thresh", type=float, default=45.0)
    args = ap.parse_args()

    df = pd.read_csv(args.parsed)

    # TNG features
    tng_ngram, tng_count = [], []
    for hyp in df["hyp"].astype(str).tolist():
        ng, c = top_ngram_count(hyp, n=args.tng_n)
        tng_ngram.append(ng); tng_count.append(c)
    df["tng_ngram"] = tng_ngram
    df["tng_count"] = tng_count

    # Global RT
    rt = repeated_targets_ratio(df["hyp"].astype(str).tolist())
    df["rt_global"] = rt

    # CHRF2 (corpus-level broadcast)
    if args.compute_chrf:
        try:
            chrf_val = chrf2(df["hyp"].astype(str).tolist(), df["ref"].astype(str).tolist())
        except Exception:
            chrf_val = np.nan
        df["chrf2"] = chrf_val
    else:
        df["chrf2"] = np.nan

    # COMET (optional)
    if args.comet_model:
        try:
            use_ref = ("qe" not in args.comet_model.lower())
            scores = comet_scores(
                df["src"].astype(str).tolist(),
                df["hyp"].astype(str).tolist(),
                df["ref"].astype(str).tolist() if use_ref else None,
                model_name=args.comet_model
            )
            df["comet"] = scores
        except Exception:
            df["comet"] = np.nan
    else:
        df["comet"] = np.nan

    # Merge human annotations nếu có
    if args.annot and os.path.exists(args.annot):
        ann = pd.read_csv(args.annot)
        if set(["id","is_hallu"]).issubset(ann.columns):
            df = df.merge(ann[["id","is_hallu"]], on="id", how="left")
            df["label"] = df["is_hallu"]
        else:
            print("[WARN] Annotation file missing required columns; ignoring.")
            df["label"] = np.nan
    else:
        df["label"] = np.nan

    # Dán nhãn rule-based nếu thiếu nhãn người gán
    mask = df["label"].isna()
    df.loc[mask, "label"] = df[mask].apply(
        lambda r: rule_label(r, args.tng_thresh, args.seqlog_thresh, args.comet_thresh, args.chrf_thresh),
        axis=1
    ).astype(int)

    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
