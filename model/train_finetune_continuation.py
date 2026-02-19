#!/usr/bin/env python3
"""Continuation fine-tune: Cstore_V1.0.1 → Cstore_V1.0.2 on updated csdDataset_small."""
import math
import csv
import re
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

FIG_DATA_DIR = Path("Fig_Data")
FIG_DATA_DIR.mkdir(exist_ok=True)

# ========== Continuation fine-tune from Cstore_V1.0.1 ==========
TRANSFER_FROM = "checkpoints/Cstore_V1.0.1/best"
MAX_LENGTH = 512
NUM_EPOCHS = 20  # fewer epochs to avoid overfitting
LEARNING_RATE = 3e-5  # lower LR to preserve learned knowledge
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
TRAIN_RATIO = 0.8
EVAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
SAVE_TOTAL_LIMIT = 3
OUTPUT_DIR = "checkpoints/Cstore_V1.0.2"


def main():
    # Data — updated csdDataset_small (with new CSD files)
    csd_dir = Path("../Dataset/csdDataset_small")
    if not csd_dir.exists():
        csd_dir = Path.cwd().parent / "Dataset" / "csdDataset_small"

    csd_file_list = sorted(csd_dir.glob("*.csd"))
    csd_text_list = []
    for f in csd_file_list:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                csd_text_list.append(content)
        except Exception as e:
            print(f"Skip {f.name}: {e}")

    print(f"Dataset (updated): {len(csd_file_list)} files, {len(csd_text_list)} documents")
    if not csd_text_list:
        raise FileNotFoundError(f"No .csd files in {csd_dir}")

    # Tokenizer from latest model
    tokenizer = GPT2Tokenizer.from_pretrained(TRANSFER_FROM)
    tokenized = tokenizer(csd_text_list, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    full_dataset = Dataset.from_dict(tokenized)

    _split1 = full_dataset.train_test_split(test_size=(EVAL_RATIO + TEST_RATIO), seed=SEED)
    train_dataset = _split1["train"]
    _remaining = _split1["test"]
    _split2 = _remaining.train_test_split(test_size=TEST_RATIO / (EVAL_RATIO + TEST_RATIO), seed=SEED)
    eval_dataset = _split2["train"]
    test_dataset = _split2["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")

    # Dataset stats → Fig_Data/Cstore_V1.0.2_dataset_stats.csv
    perm = np.random.RandomState(SEED).permutation(len(csd_text_list))
    n_train_split = int(len(csd_text_list) * TRAIN_RATIO)
    train_indices = perm[:n_train_split]
    train_texts = [csd_text_list[i] for i in train_indices]
    lengths = np.array([len(tokenizer.encode(t, add_special_tokens=False)) for t in train_texts])

    def has_pair(text, o, c):
        return o in text and c in text

    n_train = len(train_texts)
    has_synth = sum(1 for t in train_texts if has_pair(t, "<CsoundSynthesizer>", "</CsoundSynthesizer>"))
    has_instr = sum(1 for t in train_texts if has_pair(t, "<CsInstruments>", "</CsInstruments>"))
    has_score = sum(1 for t in train_texts if has_pair(t, "<CsScore>", "</CsScore>"))
    has_all = sum(
        1
        for t in train_texts
        if has_pair(t, "<CsoundSynthesizer>", "</CsoundSynthesizer>")
        and has_pair(t, "<CsInstruments>", "</CsInstruments>")
        and has_pair(t, "<CsScore>", "</CsScore>")
    )
    ext_pattern = re.compile(r"\b(diskin2?|soundin|mp3in|#include|fini?)\b|ftgen|GEN01", re.I)
    ext_count = sum(1 for t in train_texts if ext_pattern.search(t))

    ds_rows = [
        ("Category", "Item", "Value", "Source", "Notes"),
        (f"# ====== 1. TOKEN LENGTH (train split, N={n_train}) ======", "", "", "", ""),
        ("token_length", "split", f"train (N={n_train})", "seed=42 80/10/10", ""),
        ("token_length", "mean", f"{lengths.mean():.1f}", "computed", ""),
        ("token_length", "median (p50)", f"{np.percentile(lengths, 50):.0f}", "numpy.percentile", ""),
        ("token_length", "p90", f"{np.percentile(lengths, 90):.0f}", "numpy.percentile", ""),
        ("token_length", "max", f"{lengths.max()}", "numpy.max", ""),
        ("token_length", "min", f"{lengths.min()}", "numpy.min", ""),
        ("# ====== 2. TRUNCATION ======", "", "", "", ""),
        ("truncation", "max_length_used", "512", "MAX_LENGTH", ""),
        ("truncation", "samples_exceeding_512", f"{np.sum(lengths > 512)}", "np.sum(lengths > 512)", ""),
        ("truncation", "truncation_ratio", f"{100*np.sum(lengths > 512)/n_train:.1f}%", "", ""),
        ("# ====== 3. STRUCTURAL TAG COVERAGE ======", "", "", "", ""),
        ("struct_tags", "has_<CsoundSynthesizer>_pair", f"{has_synth} ({100*has_synth/n_train:.1f}%)", "string match", ""),
        ("struct_tags", "has_<CsInstruments>_pair", f"{has_instr} ({100*has_instr/n_train:.1f}%)", "string match", ""),
        ("struct_tags", "has_<CsScore>_pair", f"{has_score} ({100*has_score/n_train:.1f}%)", "string match", ""),
        ("struct_tags", "has_all_3_pairs (fully legal)", f"{has_all} ({100*has_all/n_train:.1f}%)", "", ""),
        ("# ====== 4. EXTERNAL FILE DEPENDENCY ======", "", "", "", ""),
        ("ext_file_dep", "any_external_file_ref", f"{ext_count} ({100*ext_count/n_train:.1f}%)", "regex", ""),
        ("# ====== 5. CONTINUATION FINE-TUNE ======", "", "", "", ""),
        ("transfer", "pretrained_checkpoint", TRANSFER_FROM, "", "Cstore_V1.0.1 best"),
    ]
    with open(FIG_DATA_DIR / "Cstore_V1.0.2_dataset_stats.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(ds_rows)
    print(f"[CSV] Dataset stats → {FIG_DATA_DIR / 'Cstore_V1.0.2_dataset_stats.csv'}")

    # Model — load from Cstore_V1.0.1
    print(f"Loading from {TRANSFER_FROM}")
    model = GPT2LMHeadModel.from_pretrained(TRANSFER_FROM)
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.n_layer} layers, {config.n_embd} hidden, {total_params:,} params")

    # Train
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    import torch

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EVAL_LOG_EVERY = 25
    steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM_STEPS))
    save_steps = max(EVAL_LOG_EVERY, (steps_per_epoch // EVAL_LOG_EVERY) * EVAL_LOG_EVERY)
    total_steps_est = steps_per_epoch * NUM_EPOCHS
    print(f"Steps/epoch: {steps_per_epoch}, Total: ~{total_steps_est}, Save every: {save_steps}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=EVAL_LOG_EVERY,
        eval_strategy="steps",
        eval_steps=EVAL_LOG_EVERY,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    best_dir = Path(OUTPUT_DIR) / "best"
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nBest model saved → {best_dir}")

    # ========== Export all Fig_Data CSVs ==========
    actual_total_steps = trainer.state.global_step
    hyper_rows = [
        ("model", "architecture", "GPT-2 (continuation from Cstore_V1.0.1)"),
        ("model", "pretrained_checkpoint", TRANSFER_FROM),
        ("model", "n_layer", config.n_layer),
        ("model", "n_embd (hidden_size)", config.n_embd),
        ("model", "n_head", config.n_head),
        ("model", "n_positions (max_length)", config.n_positions),
        ("model", "vocab_size", config.vocab_size),
        ("model", "total_params", total_params),
        ("model", "trainable_params", trainable_params),
        ("training", "batch_size_per_device", BATCH_SIZE),
        ("training", "gradient_accumulation_steps", GRAD_ACCUM_STEPS),
        ("training", "effective_batch_size", BATCH_SIZE * GRAD_ACCUM_STEPS),
        ("training", "learning_rate", LEARNING_RATE),
        ("training", "lr_scheduler", "cosine"),
        ("training", "warmup_ratio", WARMUP_RATIO),
        ("training", "weight_decay", WEIGHT_DECAY),
        ("training", "num_epochs", NUM_EPOCHS),
        ("training", "total_steps", actual_total_steps),
        ("training", "fp16", torch.cuda.is_available()),
        ("training", "device", device),
        ("data", "dataset", "csdDataset_small"),
        ("data", "total_documents", len(csd_text_list)),
        ("data", "train_size", len(train_dataset)),
        ("data", "eval_size", len(eval_dataset)),
        ("data", "test_size", len(test_dataset)),
        ("data", "train_ratio", TRAIN_RATIO),
        ("data", "eval_ratio", EVAL_RATIO),
        ("data", "test_ratio", TEST_RATIO),
        ("data", "seed", SEED),
        ("data", "max_length", MAX_LENGTH),
    ]
    with open(FIG_DATA_DIR / "Cstore_V1.0.2_hyperparams.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Parameter", "Value"])
        w.writerows(hyper_rows)
    print(f"[CSV] Hyperparameters → {FIG_DATA_DIR / 'Cstore_V1.0.2_hyperparams.csv'}")

    log = trainer.state.log_history
    step_data = {}
    for entry in log:
        s = entry.get("step")
        if s is None:
            continue
        if s not in step_data:
            step_data[s] = {}
        if "loss" in entry:
            step_data[s]["Training_Loss"] = round(entry["loss"], 6)
        if "eval_loss" in entry:
            step_data[s]["Eval_Loss"] = round(entry["eval_loss"], 6)
        if "learning_rate" in entry:
            step_data[s]["Learning_Rate"] = entry["learning_rate"]
        if "epoch" in entry and "Epoch" not in step_data[s]:
            step_data[s]["Epoch"] = round(entry["epoch"], 6)

    log_rows = []
    for s in sorted(step_data.keys()):
        d = step_data[s]
        log_rows.append(
            {
                "Step": s,
                "Epoch": d.get("Epoch", ""),
                "Training_Loss": d.get("Training_Loss", ""),
                "Eval_Loss": d.get("Eval_Loss", ""),
                "Learning_Rate": d.get("Learning_Rate", ""),
            }
        )
    with open(FIG_DATA_DIR / "Cstore_V1.0.2_training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Step", "Epoch", "Training_Loss", "Eval_Loss", "Learning_Rate"])
        w.writeheader()
        w.writerows(log_rows)
    print(f"[CSV] Training log → {FIG_DATA_DIR / 'Cstore_V1.0.2_training_log.csv'}  ({len(log_rows)} rows)")

    best_step, best_eval_loss = None, float("inf")
    for entry in log:
        if "eval_loss" in entry and entry["eval_loss"] < best_eval_loss:
            best_eval_loss = entry["eval_loss"]
            best_step = entry.get("step")
    with open(FIG_DATA_DIR / "Cstore_V1.0.2_best_checkpoint.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Best_Step", "Best_Eval_Loss"])
        w.writerow([best_step, round(best_eval_loss, 6)])
    print(f"[CSV] Best checkpoint → {FIG_DATA_DIR / 'Cstore_V1.0.2_best_checkpoint.csv'}  (step={best_step}, eval_loss={best_eval_loss:.6f})")

    paper_rows = [
        ("Category", "Item", "Value", "Source", "Notes"),
        ("# ====== 1. DATASET ======", "", "", "", ""),
        ("dataset", "dataset_path", str(csd_dir.resolve()), "filesystem", ""),
        ("dataset", "dataset_name", "csdDataset_small", "", "curated subset (updated)"),
        ("dataset", "total_csd_files", len(csd_file_list), "", ""),
        ("dataset", "total_documents", len(csd_text_list), "", ""),
        ("# ====== 2. CONTINUATION FINE-TUNE ======", "", "", "", ""),
        ("transfer", "pretrained_checkpoint", TRANSFER_FROM, "", "Cstore_V1.0.1 best"),
        ("transfer", "fine_tune_dataset", "csdDataset_small", "", "updated with new CSD files"),
        ("# ====== 3. DATA SPLIT ======", "", "", "", ""),
        ("split", "method", "HuggingFace train_test_split (80/10/10)", "", ""),
        ("split", "seed", SEED, "", ""),
        ("split", "train_size", len(train_dataset), "", ""),
        ("split", "eval_size", len(eval_dataset), "", ""),
        ("split", "test_size", len(test_dataset), "", ""),
        ("# ====== 4. MODEL ======", "", "", "", ""),
        ("model", "architecture", "GPT-2 (continuation)", "", ""),
        ("model", "checkpoint", OUTPUT_DIR + "/best", "", "Cstore_V1.0.2"),
        ("model", "n_layer", config.n_layer, "", ""),
        ("model", "n_embd", config.n_embd, "", ""),
        ("model", "total_params", total_params, "", ""),
        ("# ====== 5. TRAINING ======", "", "", "", ""),
        ("training", "num_epochs", NUM_EPOCHS, "", ""),
        ("training", "learning_rate", LEARNING_RATE, "", "continuation LR"),
        ("training", "batch_size", BATCH_SIZE, "", ""),
        ("training", "gradient_accumulation_steps", GRAD_ACCUM_STEPS, "", ""),
        ("training", "best_step", best_step, "", ""),
        ("training", "best_eval_loss", round(best_eval_loss, 6), "", ""),
        ("# ====== 6. EVALUATION (fill after batch_eval) ======", "", "", "", ""),
        ("batch_eval", "model_checkpoint", OUTPUT_DIR + "/best", "", "run evaluate.py to populate"),
    ]
    with open(FIG_DATA_DIR / "Cstore_V1.0.2_paper_facts.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(paper_rows)
    print(f"[CSV] Paper facts → {FIG_DATA_DIR / 'Cstore_V1.0.2_paper_facts.csv'}")

    eval_rows = [
        ("Category", "Item", "Value", "Source", "Notes"),
        ("# ====== 1. BASELINE EVAL (seed=42) ======", "", "", "", ""),
        ("repeated_eval", "seed_42_struct_legal", "TBD", "evaluate.py", "fill after eval"),
        ("repeated_eval", "seed_42_render_success", "TBD", "", ""),
        ("repeated_eval", "seed_42_has_sound", "TBD", "", ""),
        ("repeated_eval", "seed_42_missing_file", "TBD", "", ""),
        ("# ====== 2. CHECKPOINT ======", "", "", "", ""),
        ("checkpoint", "name", "Cstore_V1.0.2", "", "continuation fine-tune"),
        ("checkpoint", "path", OUTPUT_DIR + "/best", "", ""),
    ]
    with open(FIG_DATA_DIR / "Cstore_V1.0.2_eval_experiments.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(eval_rows)
    print(f"[CSV] Eval experiments template → {FIG_DATA_DIR / 'Cstore_V1.0.2_eval_experiments.csv'}")

    print("\nDone. Run: python evaluate.py --checkpoint checkpoints/Cstore_V1.0.2/best --fig_prefix Cstore --num_samples 100 --seed 42")


if __name__ == "__main__":
    main()
