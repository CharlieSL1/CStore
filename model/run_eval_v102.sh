#!/bin/bash
# Generate 100 samples with Cstore_V1.0.2
cd "$(dirname "$0")"
# Ensure dependencies in same Python that runs the script
python -m pip install -q transformers torch datasets numpy soundfile 2>/dev/null || true
python evaluate.py \
  --checkpoint checkpoints/Cstore_V1.0.2/best \
  --num_samples 100 \
  --seed 42 \
  --output_dir Generated/Cstore_V1.0.2_eval \
  --fig_prefix Cstore_V1.0.2
