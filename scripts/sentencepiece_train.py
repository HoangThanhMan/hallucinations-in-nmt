#!/usr/bin/env python3
# python scripts/sentencepiece_train.py

import os
import sys
import sentencepiece as spm

def train_spm(input_file, model_prefix, vocab_size=32000, coverage=0.9995):
    while True:
        try:
            spm.SentencePieceTrainer.Train(
                f"--input={input_file} "
                f"--model_prefix={model_prefix} "
                f"--vocab_size={vocab_size} "
                f"--model_type=unigram "
                f"--character_coverage={coverage}"
            )
            print(f"[OK] Trained {model_prefix} with vocab_size={vocab_size}")
            break
        except RuntimeError as e:
            if "Vocabulary size too high" in str(e):
                vocab_size = vocab_size // 2
                print(f"[WARN] Vocab size too big, retrying with {vocab_size}")
            else:
                raise e

def encode_file(model_file, in_file, out_file):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    with open(in_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            tokens = sp.encode(line.strip(), out_type=str)
            fout.write(" ".join(tokens) + "\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python sentencepiece_train.py SRC TGT")
        sys.exit(1)

    SRC, TGT = sys.argv[1], sys.argv[2]

    DATA_DIR = "data"
    MODEL_DIR = "sentencepiece_models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train SPM cho SRC và TGT
    for lang in [SRC, TGT]:
        input_file = f"{DATA_DIR}/train.{lang}"
        model_prefix = f"{MODEL_DIR}/spm_{lang}"
        train_spm(input_file, model_prefix)

    # Encode cho train/valid/test
    for split in ["train", "valid", "test"]:
        for lang in [SRC, TGT]:
            model_file = f"{MODEL_DIR}/spm_{lang}.model"
            in_file = f"{DATA_DIR}/{split}.{lang}"
            out_file = f"{DATA_DIR}/{split}.spm.{lang}"
            print(f"[INFO] Encoding {in_file} -> {out_file}")
            encode_file(model_file, in_file, out_file)

    print("[DONE]")

if __name__ == "__main__":
    main()
