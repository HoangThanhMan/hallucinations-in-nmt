#!/usr/bin/env bash
# Usage: bash scripts/train_fairseq.sh src tgt
set -euo pipefail
SRC=${1:-src}
TGT=${2:-tgt}

EXP=checkpoints
mkdir -p ${EXP}

fairseq-train data-bin \
    --arch transformer_iwslt_de_en \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq 2 \
    --max-epoch 10 \
    --save-dir ${EXP} \
    --fp16 \
    --log-interval 100

echo "[DONE]"
