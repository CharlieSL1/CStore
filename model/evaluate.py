#!/usr/bin/env python3
"""
Batch evaluation for CStore model.
Generates N samples, runs Csound on each, and produces eval_report.json.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Optional: soundfile for WAV analysis (more accurate than wave)
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROMPT = "<CsoundSynthesizer>"
MAX_NEW_TOKENS = 400
MIN_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_P = 0.9
TIMEOUT_SEC = 15
RMS_THRESHOLD = 0.0001
MAX_SAMPLE_THRESHOLD = 1e-6


def has_all_tags(text: str) -> bool:
    """Check if text has all 3 CSD structural tag pairs."""
    pairs = [
        ("<CsoundSynthesizer>", "</CsoundSynthesizer>"),
        ("<CsInstruments>", "</CsInstruments>"),
        ("<CsScore>", "</CsScore>"),
    ]
    return all(open_t in text and close_t in text for open_t, close_t in pairs)


def classify_failure(stderr: str) -> str:
    """Classify Csound failure from stderr."""
    stderr_lower = stderr.lower()
    if any(kw in stderr_lower for kw in ["cannot open", "not found", "no such file"]):
        return "missing_file"
    if "syntax error" in stderr_lower or "error" in stderr_lower:
        return "syntax_error"
    return "other_error"


def wav_has_sound(wav_path: Path) -> bool:
    """Check if WAV has audible content (RMS > threshold, max|sample| > threshold)."""
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return False
    try:
        if HAS_SOUNDFILE:
            data, sr = sf.read(str(wav_path), dtype="float64")
            if data.size == 0:
                return False
            rms = (data ** 2).mean() ** 0.5
            max_abs = abs(data).max()
            return rms > RMS_THRESHOLD and max_abs > MAX_SAMPLE_THRESHOLD
        else:
            import wave
            import numpy as np
            with wave.open(str(wav_path), "rb") as w:
                n = w.getnframes()
                if n == 0:
                    return False
                data = w.readframes(n)
            samples = np.frombuffer(data, dtype=np.int16)
            max_abs = np.abs(samples).max() / 32768.0
            rms = np.sqrt((samples.astype(np.float64) / 32768) ** 2).mean()
            return rms > RMS_THRESHOLD and max_abs > MAX_SAMPLE_THRESHOLD
    except Exception:
        return False


def render_csd(csd_path: Path, wav_path: Path, timeout: int = TIMEOUT_SEC):
    """
    Run Csound on CSD file. Return (success, failure_type).
    failure_type: "" if success, else "missing_file"|"syntax_error"|"other_error"|"timeout"
    """
    csd_path = csd_path.resolve()
    wav_path = wav_path.resolve()
    try:
        r = subprocess.run(
            ["csound", "-W", "-d", "-m0", "-o", str(wav_path), str(csd_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(csd_path.parent),
        )
        if r.returncode != 0:
            return False, classify_failure(r.stderr or "")
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            return False, "other_error"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "csound_not_found"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/Cstore_V1.0.1/best", help="Model checkpoint path")
    parser.add_argument("--output_dir", default="Generated/Cstore_eval", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fig_prefix", default="Cstore", help="Prefix for Fig_Data CSV")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {ckpt}")
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt)
    model = GPT2LMHeadModel.from_pretrained(ckpt)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    torch.manual_seed(args.seed)
    samples = []

    start_time = time.time()
    for i in range(args.num_samples):
        inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=False)
        if "</CsoundSynthesizer>" in text:
            text = text.split("</CsoundSynthesizer>")[0] + "</CsoundSynthesizer>\n"
        text = text[:2048]

        struct_legal = has_all_tags(text)
        token_len = len(out[0])

        csd_path = run_dir / f"sample_{i+1:03d}.csd"
        wav_path = run_dir / f"sample_{i+1:03d}.wav"
        csd_path.write_text(text, encoding="utf-8")

        render_ok, fail_type = render_csd(csd_path, wav_path)
        has_sound = wav_has_sound(wav_path) if render_ok else False

        if render_ok and not has_sound:
            fail_type = "silent_audio"

        rec = {
            "sample_id": i + 1,
            "struct_legal": bool(struct_legal),
            "render_success": bool(render_ok),
            "has_sound": bool(has_sound),
            "failure_type": str(fail_type) if not render_ok else "",
            "token_length": token_len,
            "has_eos": "<|endoftext|>" in text,
            "has_synth_close": "</CsoundSynthesizer>" in text,
            "has_instr_close": "</CsInstruments>" in text,
            "has_score_close": "</CsScore>" in text,
        }
        samples.append(rec)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.num_samples} done")

    elapsed = time.time() - start_time

    n = args.num_samples
    struct_legal = sum(1 for s in samples if s["struct_legal"])
    render_success = sum(1 for s in samples if s["render_success"])
    has_sound = sum(1 for s in samples if s["has_sound"])
    missing_file = sum(1 for s in samples if s["failure_type"] == "missing_file")
    syntax_error = sum(1 for s in samples if s["failure_type"] == "syntax_error")
    other_error = sum(1 for s in samples if s["failure_type"] == "other_error")
    timeout = sum(1 for s in samples if s["failure_type"] == "timeout")
    silent_audio = sum(1 for s in samples if s["failure_type"] == "silent_audio")
    hit_token_cap = sum(1 for s in samples if s["token_length"] >= 401)
    has_eos = sum(1 for s in samples if s["has_eos"])

    token_lengths = [s["token_length"] for s in samples]
    token_mean = sum(token_lengths) / n if n else 0
    token_p90 = sorted(token_lengths)[int(n * 0.9)] if n else 0

    report = {
        "config": {
            "model_checkpoint": str(ckpt),
            "seed": args.seed,
            "num_samples": n,
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "summary": {
            "struct_legal": f"{struct_legal}/{n} = {100*struct_legal/n:.1f}%",
            "render_success": f"{render_success}/{n} = {100*render_success/n:.1f}%",
            "has_sound": f"{has_sound}/{n} = {100*has_sound/n:.1f}%",
            "missing_file": missing_file,
            "syntax_error": syntax_error,
            "other_error": other_error,
            "timeout": timeout,
            "silent_audio": silent_audio,
            "hit_token_cap": f"{hit_token_cap}/{n} = {100*hit_token_cap/n:.1f}%",
            "has_eos": f"{has_eos}/{n} = {100*has_eos/n:.1f}%",
            "token_length_mean": token_mean,
            "token_length_p90": token_p90,
            "elapsed_sec": round(elapsed, 1),
        },
        "per_sample": samples,
    }

    report_path = run_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nEval complete: {report_path}")
    print(f"  struct_legal: {struct_legal}/{n} = {100*struct_legal/n:.1f}%")
    print(f"  render_success: {render_success}/{n} = {100*render_success/n:.1f}%")
    print(f"  has_sound: {has_sound}/{n} = {100*has_sound/n:.1f}%")
    print(f"  missing_file: {missing_file}, syntax_error: {syntax_error}, other: {other_error}, timeout: {timeout}, silent: {silent_audio}")
    print(f"  elapsed: {elapsed:.1f}s")

    # Update Fig_Data/{fig_prefix}_eval_experiments.csv
    fig_dir = Path("Fig_Data")
    if fig_dir.exists():
        eval_path = fig_dir / f"{args.fig_prefix}_eval_experiments.csv"
        rows = [
            ["Category", "Item", "Value", "Source", "Notes"],
            ["# ====== 1. BASELINE EVAL (seed=42) ======", "", "", "", ""],
            ["repeated_eval", "seed_42_struct_legal", f"{struct_legal}/{n} = {100*struct_legal/n:.1f}%", "evaluate.py", ""],
            ["repeated_eval", "seed_42_render_success", f"{render_success}/{n} = {100*render_success/n:.1f}%", "", ""],
            ["repeated_eval", "seed_42_has_sound", f"{has_sound}/{n} = {100*has_sound/n:.1f}%", "", ""],
            ["repeated_eval", "seed_42_missing_file", f"{missing_file}/{n}", "", ""],
            ["repeated_eval", "seed_42_silent_audio", f"{silent_audio}/{n}", "", ""],
            ["repeated_eval", "seed_42_syntax_error", f"{syntax_error}/{n}", "", ""],
            ["repeated_eval", "seed_42_other_error", f"{other_error}/{n}", "", ""],
            ["repeated_eval", "seed_42_timeout", f"{timeout}/{n}", "", ""],
            ["repeated_eval", "seed_42_hit_token_cap", f"{hit_token_cap}/{n} = {100*hit_token_cap/n:.1f}%", "", ""],
            ["repeated_eval", "seed_42_has_eos", f"{has_eos}/{n} = {100*has_eos/n:.1f}%", "", ""],
            ["repeated_eval", "seed_42_elapsed_sec", f"{elapsed:.1f}s", "", ""],
            ["# ====== 2. CHECKPOINT ======", "", "", "", ""],
            ["checkpoint", "name", Path(ckpt).parent.name, "", ""],
            ["checkpoint", "path", str(ckpt), "", ""],
            ["checkpoint", "eval_report_path", str(report_path), "", ""],
        ]
        import csv
        with open(eval_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(rows)
        print(f"  Updated {eval_path}")

    return report_path

if __name__ == "__main__":
    main()
