# CStore Model

## Checkpoints

> **Note:** Checkpoints (~130MB each) are not in the repo. Download from [Releases](https://github.com/CharlieSL1/CStore/releases) or place your trained checkpoints at `model/checkpoints/`.

| Checkpoint | Paper | Step | Val Loss | Struct | Render | Sound |
|------------|-------|------|----------|--------|--------|-------|
| `checkpoints/Cstore_V1.0.0/best` | Baseline (full corpus) | 26,650 | 0.387 | 61% | 32% | 19% |
| `checkpoints/Cstore_V1.0.1/best` | Expert-curated fine-tune (FT1) | 500 | 0.360 | 73% | 56% | 54% |
| `checkpoints/Cstore_V1.0.2/best` | Continuation fine-tune (FT2) | 375 | 0.320 | 72% | 53% | 50% |

---

## Training Scripts

| Script | Input | Output |
|--------|-------|--------|
| `train_finetune_expert.py` | Cstore_V1.0.0 + csdDataset_small | Cstore_V1.0.1 |
| `train_finetune_continuation.py` | Cstore_V1.0.1 + csdDataset_small | Cstore_V1.0.2 |
| `train_prefix_continuation.py` | Cstore_V1.0.1 + csdDataset_small | continuation_ckpt |

**Dataset**: Place `csdDataset_small` (expert-curated .csd files) at `../Dataset/csdDataset_small` (i.e. `CStore/Dataset/csdDataset_small`).

```bash
cd model
python train_finetune_expert.py       # Fine-tune V1.0.0 → V1.0.1
python train_finetune_continuation.py # Fine-tune V1.0.1 → V1.0.2
```

---

## Evaluation & Generation

```bash
cd model
python evaluate.py --checkpoint checkpoints/Cstore_V1.0.1/best --num_samples 100 --seed 42
python generate.py   # Edit INPUT_CSD_PATH or use from_scratch mode
```

---

## Load Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "checkpoints/Cstore_V1.0.1/best"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
```
