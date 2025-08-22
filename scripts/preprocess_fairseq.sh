#!/usr/bin/env bash
# Usage: bash scripts/preprocess_fairseq.sh src tgt

set -euo pipefail
SRC=${1:-src}
TGT=${2:-tgt}

mkdir -p data-bin

# Dùng .spm.* nếu đã encode, mặc định USE_SPM=1
USE_SPM=${USE_SPM:-1}
if [[ "$USE_SPM" == "1" ]]; then
    TRAIN_PREF=data/train.spm
    VALID_PREF=data/valid.spm
    TEST_PREF=data/test.spm
else
    TRAIN_PREF=data/train
    VALID_PREF=data/valid
    TEST_PREF=data/test
fi

fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TRAIN_PREF --validpref $VALID_PREF --testpref $TEST_PREF \
    --destdir data-bin --workers 4

echo "[DONE]"
