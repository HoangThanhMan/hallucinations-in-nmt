#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Setting up environment for NMT Hallucination project..."

# ===============================
# 1. Cập nhật pip + công cụ cơ bản
# ===============================
pip install --upgrade wheel setuptools

# ===============================
# 2. Cài torch phù hợp Colab
# Colab thường đã có torch + cuda tương thích,
# nếu muốn cố định phiên bản thì uncomment dòng bên dưới:
# pip install torch==2.3.0 torchvision torchaudio
# ===============================

# ===============================
# 3. Cài fairseq & các thư viện cần thiết
# ===============================
pip install fairseq==0.12.2

# Metric & preprocessing
pip install sacrebleu sentencepiece

# COMET để đánh giá dịch máy
pip install unbabel-comet==2.2.1

# ML + xử lý dữ liệu
pip install pandas numpy scikit-learn joblib tqdm matplotlib

echo "[OK] Environment ready!"
