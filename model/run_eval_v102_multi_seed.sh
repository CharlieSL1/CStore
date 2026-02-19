#!/bin/bash
# Generate 100 samples total: 10 seeds × 10 samples each (Cstore_V1.0.2)
cd "$(dirname "$0")"
python -m pip install -q transformers torch datasets numpy soundfile 2>/dev/null || true

SEEDS=(42 123 456 789 111 222 333 444 555 666)
for seed in "${SEEDS[@]}"; do
  echo "=== Seed $seed (10 samples) ==="
  python evaluate.py \
    --checkpoint checkpoints/Cstore_V1.0.2/best \
    --num_samples 10 \
    --seed "$seed" \
    --output_dir Generated/Cstore_V1.0.2_eval \
    --fig_prefix Cstore_V1.0.2
done
echo "Done. 10 seeds × 10 samples = 100 total"
