#!/usr/bin/env python3
"""
Continuation fine-tuning: train model to continue from user-provided CSD prefix.
Loss is computed only on the suffix (continuation) tokens, not the prefix.
"""
import math
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

# ======================== CONFIG ========================
TRANSFER_FROM = "checkpoints/Cstore_V1.0.1/best"
MAX_LENGTH = 512
TRUNCATE_RATIO = 0.65  # same as generate.py for consistency
NUM_EPOCHS = 8
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
TRAIN_RATIO = 0.9
EVAL_RATIO = 0.1
SEED = 42
SAVE_TOTAL_LIMIT = 2
OUTPUT_DIR = "checkpoints/continuation_ckpt"
MIN_SUFFIX_TOKENS = 30  # skip if suffix too short
# ========================================================


def _extract_first_n_instr_impl(csd_text, n=1, truncate_ratio=1.0):
    """Return (prefix_lines_end_index, prefix_text). prefix = lines[:end_idx]."""
    lines = csd_text.split("\n")
    instr_blocks = []
    i = 0
    while i < len(lines):
        if re.match(r"\s*instr\b", lines[i]):
            start = i
            i += 1
            while i < len(lines):
                if re.match(r"\s*endin\b", lines[i]):
                    instr_blocks.append((start, i))
                    i += 1
                    break
                i += 1
            if len(instr_blocks) >= n:
                break
        else:
            i += 1
    if not instr_blocks:
        return len(lines), csd_text
    last_start, last_end = instr_blocks[n - 1]
    if truncate_ratio >= 1.0:
        cut_at = last_end + 1
    else:
        n_instr_lines = last_end - last_start + 1
        keep = max(2, int(n_instr_lines * truncate_ratio))
        cut_at = last_start + keep
    prefix = "\n".join(lines[:cut_at]) + "\n"
    return cut_at, prefix


def build_continuation_pairs(csd_text_list, truncate_ratio=0.65):
    """Build (prefix, suffix) pairs. Skip external-file deps and too-short suffixes."""
    ext_pattern = re.compile(r"\b(diskin2?|soundin|mp3in|#include|fini?)\b|ftgen|GEN01", re.I)
    pairs = []
    for text in csd_text_list:
        if ext_pattern.search(text):
            continue
        if "</CsoundSynthesizer>" in text:
            text = text.split("</CsoundSynthesizer>")[0] + "</CsoundSynthesizer>\n"
        cut_at, prefix = _extract_first_n_instr_impl(text, n=1, truncate_ratio=truncate_ratio)
        lines = text.split("\n")
        suffix = "\n".join(lines[cut_at:])
        if not suffix.strip() or len(suffix.strip()) < 50:
            continue
        pairs.append((prefix, suffix))
    return pairs


def tokenize_with_labels(prefix, suffix, tokenizer, max_length=512):
    """Tokenize prefix+suffix, set labels=-100 for prefix tokens."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if len(prefix_ids) >= max_length:
        return None
    suffix_len = min(len(suffix_ids), max_length - len(prefix_ids))
    if suffix_len <= 0:
        return None
    suffix_ids = suffix_ids[:suffix_len]
    total_ids = prefix_ids + suffix_ids
    labels = [-100] * len(prefix_ids) + suffix_ids
    return {
        "input_ids": total_ids,
        "labels": labels,
        "attention_mask": [1] * len(total_ids),
    }


def main():
    script_dir = Path(__file__).parent
    csd_dir = script_dir.parent / "Dataset" / "csdDataset_small"
    if not csd_dir.exists():
        csd_dir = Path.cwd().parent / "Dataset" / "csdDataset_small"
    assert csd_dir.exists(), f"Dataset not found: {csd_dir}"

    csd_files = sorted(csd_dir.glob("*.csd"))
    csd_text_list = []
    for f in csd_files:
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
            if t.strip():
                csd_text_list.append(t)
        except Exception as e:
            print(f"Skip {f.name}: {e}")

    print(f"Loaded {len(csd_text_list)} CSD files")

    pairs = build_continuation_pairs(csd_text_list, truncate_ratio=TRUNCATE_RATIO)
    print(f"Built {len(pairs)} continuation pairs (truncate={TRUNCATE_RATIO})")

    tokenizer = GPT2Tokenizer.from_pretrained(TRANSFER_FROM)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    for prefix, suffix in pairs:
        item = tokenize_with_labels(prefix, suffix, tokenizer, MAX_LENGTH)
        if item is None:
            continue
        n_suffix = len([x for x in item["labels"] if x != -100])
        if n_suffix < MIN_SUFFIX_TOKENS:
            continue
        input_ids_list.append(item["input_ids"])
        labels_list.append(item["labels"])
        attention_mask_list.append(item["attention_mask"])

    if not input_ids_list:
        raise ValueError("No valid samples after tokenization")

    print(f"Valid samples: {len(input_ids_list)}")

    full_dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    })

    split = full_dataset.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    model = GPT2LMHeadModel.from_pretrained(TRANSFER_FROM)
    config = model.config
    print(f"Model: {config.n_layer} layers, {config.n_embd} hidden")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EVAL_EVERY = 25
    steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM_STEPS))
    save_steps = ((steps_per_epoch + EVAL_EVERY - 1) // EVAL_EVERY) * EVAL_EVERY
    save_steps = max(EVAL_EVERY, save_steps)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=EVAL_EVERY,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=SEED,
    )

    def collate_fn(examples):
        max_len = max(len(ex["input_ids"]) for ex in examples)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = []
        labels = []
        attention_mask = []
        for ex in examples:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_id] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

    best_dir = Path(OUTPUT_DIR) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nContinuation-finetuned model saved → {best_dir}")
    print("Use in generate.py: CHECKPOINT = 'checkpoints/continuation_ckpt/best'")


if __name__ == "__main__":
    main()
