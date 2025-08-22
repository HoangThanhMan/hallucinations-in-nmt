#!/usr/bin/env bash
# Usage: bash scripts/01_sentencepiece_train.sh src tgt

set -euo pipefail
SRC=${1:-src}
TGT=${2:-tgt}

mkdir -p data sentencepiece_models

# Train SPM
for LANG in $SRC $TGT; do
  spm_train --input=data/train.${LANG} --model_prefix=sentencepiece_models/spm_${LANG} \
            --vocab_size=32000 --model_type=unigram --character_coverage=0.9995
done

# Encode
for SPLIT in train valid test; do
  for LANG in $SRC $TGT; do
    spm_encode --model=sentencepiece_models/spm_${LANG}.model \
               --output_format=piece < data/${SPLIT}.${LANG} > data/${SPLIT}.spm.${LANG}
  done
done

echo "[OK] SentencePiece models trained and applied."
