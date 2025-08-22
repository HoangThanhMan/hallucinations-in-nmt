#!/usr/bin/env bash
# Usage: bash scripts/generate_fairseq.sh src tgt
set -euo pipefail
SRC=${1:-src}
TGT=${2:-tgt}

EXP=checkpoints
OUT=outputs
mkdir -p ${OUT}

fairseq-generate data-bin \
    --path ${EXP}/checkpoint_best.pt \
    --beam 5 --lenpen 1.0 --batch-size 128 \
    --remove-bpe=sentencepiece \
    --scoring sacrebleu > ${OUT}/generate.test.txt

python scripts/05_parse_generate.py \
    --gen ${OUT}/generate.test.txt \
    --out ${OUT}/generate.test.parsed.csv

echo "[DONE]"
