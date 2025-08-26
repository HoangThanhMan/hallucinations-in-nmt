#!/usr/bin/env python
import argparse, os, sys
import pandas as pd
import numpy as np

# Ensure local import path
sys.path.append(os.path.dirname(__file__))
from detectors import top_ngram_count, repeated_targets, chrf2, comet_scores

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

def select_examples(df, col, worst_k=250, tail_frac=0.004, tail_k=250, larger_is_worse=False):
    df = df.dropna(subset=[col])
    if df.empty:
        return pd.DataFrame()
    
    if larger_is_worse:
        df_sorted = df.sort_values(col, ascending=False)
    else:
        df_sorted = df.sort_values(col, ascending=True)

    worst = df_sorted.head(worst_k)["id"].tolist()

    n_tail = max(int(len(df_sorted) * tail_frac), tail_k)
    tail_candidates = df_sorted.head(n_tail)
    tail = tail_candidates.sample(min(tail_k, len(tail_candidates)), random_state=42)["id"].tolist()

    return list(set(worst + tail))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True, help="CSV from parse_generate.py")
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
    tng_list = []
    for src, hyp in zip(df["src"].astype(str).tolist(),df["hyp"].astype(str).tolist()):
        _, c_src = top_ngram_count(src, n=args.tng_n)
        _, c_hyp = top_ngram_count(hyp, n=args.tng_n)
        if c_hyp - c_src >= args.tng_thresh:
            tng_list.append(c_hyp - c_src)
        else:
            tng_list.append(0)
    df["tng_count"] = tng_list

    # Global RT
    rt = repeated_targets(df["hyp"].astype(str).tolist())
    rt_list = []
    for hyp in df["hyp"].astype(str).tolist():
        rt_list.append(rt.get(hyp, 0))
    df["rt"] = rt_list

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

    metrics = {
        "comet": {"larger_is_worse": False},
        "tng_count": {"larger_is_worse": True},
        "rt": {"larger_is_worse": True},
        "chrf2": {"larger_is_worse": False},
        "attn_eos": {"larger_is_worse": True},
        "attn_ignsrc": {"larger_is_worse": True},
    }

    selected_ids = set()
    for col, cfg in metrics.items():
        if col in df.columns:
            picked = select_examples(df, col,
                                     worst_k=250,
                                     tail_frac=0.004,
                                     tail_k=250,
                                     larger_is_worse=cfg["larger_is_worse"])
            print(f"[INFO] {col}: chọn {len(picked)} mẫu")
            selected_ids.update(picked)

    df_out = df[df["id"].isin(selected_ids)].copy()

    # Xuất kết quả
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote {args.out} với {len(df_out)} samples (after dedup).")

    # # Dán nhãn rule-based nếu thiếu nhãn người gán
    # mask = df["label"].isna()
    # df.loc[mask, "label"] = df[mask].apply(
    #     lambda r: rule_label(r, args.tng_thresh, args.seqlog_thresh, args.comet_thresh, args.chrf_thresh),
    #     axis=1
    # ).astype(int)

    # df.to_csv(args.out, index=False)
    # print(f"[OK] Wrote {args.out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
