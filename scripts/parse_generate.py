#!/usr/bin/env python

import csv, argparse

def parse_generate(gen_path, out_csv):
    # Parse fairseq-generate output to CSV với các cột:
    # id, src, ref, hyp, seq_logprob
    S, T, H = {}, {}, {}
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("S-"):
                idx, rest = line.split("\t", 1)
                i = int(idx[2:])
                S[i] = rest.strip()
            elif line.startswith("T-"):
                idx, rest = line.split("\t", 1)
                i = int(idx[2:])
                T[i] = rest.strip()
            elif line.startswith("H-"):
                parts = line.strip().split("\t")
                i = int(parts[0][2:])
                score = float(parts[1])
                hyp = parts[2]
                H[i] = (score, hyp)

    ids = sorted(set(S) & set(T) & set(H))
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id","src","ref","hyp","seq_logprob"])
        for i in ids:
            w.writerow([i, S[i], T[i], H[i][1], H[i][0]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parse_generate(args.gen, args.out)
