#!/usr/bin/env python3
"""
CStore Web App - Local server for generating CSD and audio.
Uses CStore V1.0.1 (best model) for generation.
"""
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add model directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from flask import Flask, jsonify, request, send_from_directory, send_file

# Model checkpoint - V1.0.1 is the best (73% struct, 56% render, 54% sound)
CHECKPOINT = PROJECT_ROOT / "model" / "checkpoints" / "Cstore_V1.0.1" / "best"
GENERATED_DIR = PROJECT_ROOT / "webapp" / "generated"
MAX_GENERATE_ATTEMPTS = 80  # Watchdog: retry until audio has audible content (RMS check)

app = Flask(__name__, static_folder="static")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Lazy load model
_model = None
_tokenizer = None


def fix_common_model_errors(text):
    """Fix common model output errors."""
    text = re.sub(r"\boutch\s+\d+\s*,\s*", "outs ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bout\s+([^;\n]+)(?=\s*[;\n])", r"outs \1, \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\$\w+", "1", text)
    return text


def repair_csd(text, original_score=None):
    """Repair incomplete CSD structure."""
    if "</CsoundSynthesizer>" in text:
        text = text.split("</CsoundSynthesizer>")[0] + "</CsoundSynthesizer>\n"
    lines = text.split("\n")
    last_instr_idx, endin_after = None, False
    complete_instrs = []
    current_instr = "1"
    for i, line in enumerate(lines):
        if re.match(r"\s*instr\b", line.strip()):
            last_instr_idx, endin_after = i, False
            m = re.match(r"\s*instr\s+(\S+)", line.strip())
            if m:
                current_instr = m.group(1)
        if re.match(r"\s*endin\b", line.strip()) and last_instr_idx is not None:
            endin_after = True
            complete_instrs.append(current_instr)
    if last_instr_idx is not None and not endin_after:
        # Incomplete instr: insert endin before </CsInstruments> or at end
        insert_at = len(lines)
        for i in range(last_instr_idx + 1, len(lines)):
            if "</CsInstruments>" in lines[i] or "<CsScore>" in lines[i]:
                insert_at = i
                break
        lines = lines[:insert_at] + ["endin"] + lines[insert_at:]
        complete_instrs.append(current_instr)
    if not complete_instrs:
        # No instr at all: inject minimal instr before </CsInstruments> if present
        insert_idx = None
        for i, line in enumerate(lines):
            if "</CsInstruments>" in line:
                insert_idx = i
                break
        if insert_idx is not None:
            inject = ["instr 1", "aout poscil 0.2, 440, 1", "outs aout, aout", "endin"]
            lines = lines[:insert_idx] + inject + lines[insert_idx:]
            complete_instrs = ["1"]
        else:
            return None
    text = "\n".join(lines)
    if "</CsInstruments>" not in text and "<CsInstruments>" in text:
        text += "\n</CsInstruments>\n"
    if "<CsScore>" not in text:
        score = original_score if original_score else f"i {complete_instrs[0]} 0 3\ne"
        text += f"\n<CsScore>\n{score}\n</CsScore>\n"
    elif "</CsScore>" not in text:
        text += "\ne\n</CsScore>\n"
    else:
        m = re.search(r"<CsScore>\s*(.*?)\s*</CsScore>", text, re.DOTALL)
        if m and not re.search(r"\be\b", m.group(1)):
            text = text.replace("</CsScore>", "\ne\n</CsScore>", 1)
    if "</CsoundSynthesizer>" not in text:
        text += "</CsoundSynthesizer>\n"
    return text


RMS_THRESHOLD = 0.0001
MAX_SAMPLE_THRESHOLD = 1e-6


def wav_has_sound(wav_path: Path) -> bool:
    """Check if WAV has audible content (RMS > threshold, max|sample| > threshold)."""
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return False
    try:
        import soundfile as sf
        data, _ = sf.read(str(wav_path), dtype="float64")
        if data.size == 0:
            return False
        rms = (data ** 2).mean() ** 0.5
        max_abs = abs(data).max()
        return rms > RMS_THRESHOLD and max_abs > MAX_SAMPLE_THRESHOLD
    except ImportError:
        import wave
        import numpy as np
        try:
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
    except Exception:
        return False


def render_csd_to_wav(csd_path: Path, wav_path: Path, timeout: int = 15) -> bool:
    """Run Csound to render CSD to WAV. Returns True only if render succeeds AND audio has sound."""
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
        if r.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 500:
            return False
        return wav_has_sound(wav_path)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def load_model():
    """Load CStore model (lazy)."""
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {CHECKPOINT}\n"
                "Download from https://github.com/CharlieSL1/CStore/releases"
            )
        _tokenizer = GPT2Tokenizer.from_pretrained(str(CHECKPOINT))
        _model = GPT2LMHeadModel.from_pretrained(str(CHECKPOINT))
        _model.eval()
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _model = _model.to(device)
    return _model, _tokenizer


def generate_one_sample(seed: int = None):
    """Generate one CSD sample, render to WAV, save both. Returns (csd_text, csd_path, wav_path) or (None, None, None)."""
    import torch

    model, tokenizer = load_model()
    device = next(model.parameters()).device

    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % (2**32)

    torch.manual_seed(seed)

    PROMPT = "<CsoundSynthesizer>"
    MAX_NEW_TOKENS = 400
    MIN_NEW_TOKENS = 100
    TEMPERATURE = 0.8
    TOP_P = 0.9

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
    raw_text = tokenizer.decode(out[0], skip_special_tokens=False)
    if "</CsoundSynthesizer>" in raw_text:
        raw_text = raw_text.split("</CsoundSynthesizer>")[0] + "</CsoundSynthesizer>\n"
    raw_text = raw_text[:2048]

    gen_text = repair_csd(fix_common_model_errors(raw_text))
    if gen_text is None:
        return None, None, None

    # Save to generated folder with unique ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = GENERATED_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    csd_path = run_dir / "output.csd"
    wav_path = run_dir / "output.wav"
    csd_path.write_text(gen_text, encoding="utf-8")

    if not render_csd_to_wav(csd_path, wav_path):
        # Remove failed run dir (watchdog: only keep successful outputs)
        try:
            shutil.rmtree(run_dir)
        except OSError:
            pass
        return None, None, None

    return gen_text, str(csd_path), str(wav_path)


def generate_until_audio_success(seed: int = None):
    """Watchdog: retry until audio renders successfully. Only returns when both CSD and WAV exist."""
    base_seed = seed if seed is not None else int(datetime.now().timestamp() * 1000) % (2**32)
    for attempt in range(1, MAX_GENERATE_ATTEMPTS + 1):
        try_seed = base_seed + attempt * 12345
        csd_text, csd_path, wav_path = generate_one_sample(seed=try_seed)
        if wav_path is not None:
            return csd_text, csd_path, wav_path
    return None, None, None


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate CSD + WAV. Watchdog: only returns when audio renders successfully."""
    try:
        data = request.get_json() or {}
        seed = data.get("seed")
        if seed is not None:
            seed = int(seed)

        csd_text, csd_path, wav_path = generate_until_audio_success(seed=seed)

        if csd_text is None or wav_path is None:
            return jsonify({
                "error": f"No valid audio after {MAX_GENERATE_ATTEMPTS} attempts. Check Csound installation."
            }), 500

        run_id = Path(csd_path).parent.name
        wav_url = f"/generated/{run_id}/output.wav"
        result = {
            "success": True,
            "csd": csd_text,
            "run_id": run_id,
            "csd_url": f"/generated/{run_id}/output.csd",
            "wav_url": wav_url,
        }
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        err = str(e)
        if not err or len(err) > 200:
            err = "Generation failed"
        return jsonify({"error": err}), 500


@app.route("/generated/<run_id>/<filename>")
def serve_generated(run_id, filename):
    """Serve generated CSD or WAV files."""
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        return "Not found", 404
    if filename not in ("output.csd", "output.wav"):
        return "Not found", 404
    path = GENERATED_DIR / run_id / filename
    if not path.exists():
        return "Not found", 404
    return send_file(path, as_attachment=False, download_name=filename)


@app.route("/api/list")
def api_list():
    """List all generated runs."""
    runs = []
    try:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        for d in sorted(GENERATED_DIR.iterdir(), reverse=True):
            if d.is_dir():
                csd = d / "output.csd"
                wav = d / "output.wav"
                runs.append({
                    "run_id": d.name,
                    "has_csd": csd.exists(),
                    "has_wav": wav.exists(),
                })
    except (FileNotFoundError, OSError):
        runs = []
    return jsonify({"runs": runs[:50]})


if __name__ == "__main__":
    print("CStore Web App - http://127.0.0.1:5000")
    print("Model: Cstore_V1.0.1 (best)")
    print("Csound required for audio rendering.")
    app.run(host="127.0.0.1", port=5000, debug=False)
