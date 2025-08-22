#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -y
sudo apt-get install python3.9 python3.9-dev python3.9-distutils -y
wget https://bootstrap.pypa.io/get-pip.py
python3.9 get-pip.py

pip install pip==24.0
python3.9 -m pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python3.9 -m pip install fairseq==0.12.2 sacremoses subword-nmt sentencepiece sacrebleu unbabel-comet==2.2.1 joblib tqdm matplotlib
python3.9 -m pip install --upgrade "numpy<2" 

echo "[DONE]"
