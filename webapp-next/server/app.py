#!/usr/bin/env python3
"""
CStore backend — Python sidecar for the Next.js console in ../app/.
Serves:
  POST /api/generate     · GET /api/list
  GET  /api/models       · GET|POST /api/model
  GET  /generated/<id>/output.{csd,wav}
The Next.js dev server (in the parent folder) proxies /api/* and /generated/*
to this process via next.config.ts.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

# This file now lives at  <repo>/webapp-next/server/app.py
# so PROJECT_ROOT must climb three levels, not two.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEBAPP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from flask import Flask, jsonify, request, send_file

# Default checkpoint. V1.0.1/best has the highest published metrics
# (73% struct / 56% render / 54% sound in the README); V1.0.2/best is the
# continuation-trained variant and is the current default.
# The active checkpoint can be swapped at runtime via /api/model or by passing
# `checkpoint` in /api/generate.
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "checkpoints" / "Cstore_V1.0.2" / "best"
MODEL_ROOT = PROJECT_ROOT / "model" / "checkpoints"
# Rendered outputs now live inside the Next.js project so everything is in one tree.
GENERATED_DIR = WEBAPP_DIR / "generated"
MAX_GENERATE_ATTEMPTS = 80  # Watchdog: retry until audio has audible content (RMS check)

# No static_folder: the UI is served by Next.js (localhost:3000), not Flask.
app = Flask(__name__)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Lazy load model + currently-active checkpoint path.
# Only one (model, tokenizer) pair is resident at a time; switching frees the
# previous pair so we don't blow up memory when users flip through V1.0.0/1/2.
_model = None
_tokenizer = None
_active_checkpoint: Path = DEFAULT_CHECKPOINT

# ---------------------------------------------------------------------------
# Console log (surfaced at GET /api/console for the web terminal).
# A ring buffer of structured events, each with a monotonic seq so clients can
# poll for "everything new since seq N". Captures csound stdout/stderr plus
# lifecycle markers (generate/attempt/render/wav-check/edit).
# ---------------------------------------------------------------------------
CONSOLE_MAX_ENTRIES = 2000
_console_lock = threading.Lock()
_console_seq = 0
_console_log: "deque[dict]" = deque(maxlen=CONSOLE_MAX_ENTRIES)
_console_started_at = time.time()


def log_console(level: str, text: str, run_id: str | None = None) -> None:
    """Append one line to the console ring buffer. Safe to call from anywhere.

    level is a short tag: "info" | "csound" | "warn" | "err" | "ok" | "sys".
    Multi-line `text` is split so every line gets its own seq and timestamp,
    which keeps the front-end rendering simple (one entry == one row).
    """
    global _console_seq
    if text is None:
        return
    now = time.time()
    with _console_lock:
        for raw_line in str(text).splitlines() or [""]:
            _console_seq += 1
            _console_log.append({
                "seq": _console_seq,
                "t": now,
                "level": level,
                "run_id": run_id,
                "text": raw_line.rstrip(),
            })


def console_snapshot(since: int = 0, limit: int = 500) -> dict:
    with _console_lock:
        latest = _console_seq
        if since <= 0:
            entries = list(_console_log)
        else:
            entries = [e for e in _console_log if e["seq"] > since]
        if len(entries) > limit:
            entries = entries[-limit:]
    return {"seq": latest, "entries": entries, "started_at": _console_started_at}


def resolve_checkpoint(raw: str | None) -> Path:
    """Resolve a user-supplied checkpoint string into a Path and validate it.

    - Empty/None falls back to the currently active checkpoint.
    - Paths under MODEL_ROOT may be given as relative labels like
      "Cstore_V1.0.1/best" for convenience.
    - Absolute paths are accepted (this server is localhost-only).
    - The path must be an existing directory containing config.json, which is
      the minimum Hugging Face transformers requires to load a model.
    """
    if not raw:
        return _active_checkpoint
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        # Try it relative to model/checkpoints first, then to the project root.
        rel_under_models = (MODEL_ROOT / candidate).resolve()
        if rel_under_models.exists():
            candidate = rel_under_models
        else:
            candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {candidate}")
    if not (candidate / "config.json").exists():
        raise ValueError(
            f"Not a valid model folder (missing config.json): {candidate}"
        )
    return candidate


def discover_checkpoints() -> list[dict]:
    """Find every folder under model/checkpoints/ that looks like a HF model.

    Returns entries sorted with each version's `best` first, then training
    checkpoints in numeric order. The frontend uses `path` (relative label)
    when round-tripping a selection back to the server.
    """
    results: list[dict] = []
    if not MODEL_ROOT.exists():
        return results

    def _numeric_suffix(name: str) -> int:
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else -1

    for version_dir in sorted(MODEL_ROOT.iterdir()):
        if not version_dir.is_dir():
            continue
        subs = [p for p in version_dir.iterdir() if p.is_dir()]
        # `best` always first, then checkpoint-* ordered by step number descending.
        subs.sort(
            key=lambda p: (0 if p.name == "best" else 1, -_numeric_suffix(p.name), p.name)
        )
        for sub in subs:
            if not (sub / "config.json").exists():
                continue
            rel = sub.relative_to(MODEL_ROOT).as_posix()
            results.append(
                {
                    "path": rel,  # e.g. "Cstore_V1.0.1/best"
                    "absolute": str(sub),
                    "family": version_dir.name,
                    "variant": sub.name,
                    "label": f"{version_dir.name} / {sub.name}",
                    "is_default": sub.resolve() == DEFAULT_CHECKPOINT.resolve(),
                }
            )
    return results


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
        ok = rms > RMS_THRESHOLD and max_abs > MAX_SAMPLE_THRESHOLD
        log_console(
            "info" if ok else "warn",
            f"wav check · rms={rms:.4e} max|x|={max_abs:.4e} → {'pass' if ok else 'silent'}",
            run_id=wav_path.parent.name,
        )
        return ok
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
    """Run Csound to render CSD to WAV. Returns True only if render succeeds AND audio has sound.

    csound output is forwarded line-by-line into the console log buffer so the
    web terminal can show what csound is actually doing (orchestra parse,
    instrument compile, score events, warnings, etc.).
    """
    csd_path = csd_path.resolve()
    wav_path = wav_path.resolve()
    run_id = csd_path.parent.name
    # -m135 asks csound to print note amps, out-of-range warnings, and sample-
    # count summaries — the useful "what happened" lines without flooding.
    cmd = ["csound", "-W", "-d", "-m135", "-o", str(wav_path), str(csd_path)]
    log_console("info", f"$ {' '.join(cmd)}", run_id=run_id)
    started = time.time()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(csd_path.parent),
        )
    except subprocess.TimeoutExpired:
        log_console("err", f"csound timed out after {timeout}s", run_id=run_id)
        return False
    except FileNotFoundError:
        log_console("err", "csound binary not found on PATH", run_id=run_id)
        return False

    # Csound writes almost everything to stderr (that's normal). Stream both.
    for line in (r.stdout or "").splitlines():
        log_console("csound", line, run_id=run_id)
    for line in (r.stderr or "").splitlines():
        log_console("csound", line, run_id=run_id)
    dt = time.time() - started

    if r.returncode != 0:
        log_console(
            "err",
            f"csound exit={r.returncode} after {dt:.2f}s",
            run_id=run_id,
        )
        return False
    if not wav_path.exists() or wav_path.stat().st_size < 500:
        log_console(
            "err",
            f"csound produced no / tiny wav ({wav_path.stat().st_size if wav_path.exists() else 0} bytes)",
            run_id=run_id,
        )
        return False
    log_console(
        "ok",
        f"csound render ok in {dt:.2f}s · {wav_path.stat().st_size} bytes",
        run_id=run_id,
    )
    return True


def load_model(checkpoint: Path | None = None):
    """Load CStore model, swapping if a different checkpoint was requested.

    The previous model/tokenizer pair is dropped before loading the new one,
    which is important on MPS/CUDA where a second model would double VRAM.
    """
    global _model, _tokenizer, _active_checkpoint
    target = (checkpoint or _active_checkpoint).resolve()

    if _model is not None and _active_checkpoint.resolve() == target:
        return _model, _tokenizer

    import gc

    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    if not target.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {target}\n"
            "Download from https://github.com/CharlieSL1/CStore/releases"
        )

    log_console("sys", f"loading checkpoint · {target}")
    # Release the previous model first so memory doesn't spike.
    if _model is not None:
        del _model
        _model = None
        del _tokenizer
        _tokenizer = None
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except (AttributeError, RuntimeError):
                pass
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokenizer = GPT2Tokenizer.from_pretrained(str(target))
    model = GPT2LMHeadModel.from_pretrained(str(target))
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    _model = model
    _tokenizer = tokenizer
    _active_checkpoint = target
    log_console("ok", f"checkpoint ready · device={device}")
    return _model, _tokenizer


def generate_one_sample(seed: int = None, checkpoint: Path | None = None):
    """Generate one CSD sample, render to WAV, save both. Returns (csd_text, csd_path, wav_path) or (None, None, None)."""
    import torch

    model, tokenizer = load_model(checkpoint)
    device = next(model.parameters()).device

    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % (2**32)

    torch.manual_seed(seed)
    t0 = time.time()
    log_console("info", f"sampling · seed={seed} · T=0.8 top_p=0.9 max_new=400")

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
        log_console("warn", "repair_csd returned None — malformed .csd, discarding")
        return None, None, None
    log_console(
        "info",
        f"sampled {len(gen_text)} chars in {time.time() - t0:.2f}s",
    )

    # Save to generated folder with unique ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = GENERATED_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    csd_path = run_dir / "output.csd"
    wav_path = run_dir / "output.wav"
    csd_path.write_text(gen_text, encoding="utf-8")
    log_console("info", f"wrote {csd_path.relative_to(WEBAPP_DIR)}", run_id=run_id)

    if not render_csd_to_wav(csd_path, wav_path):
        # Remove failed run dir (watchdog: only keep successful outputs)
        try:
            shutil.rmtree(run_dir)
        except OSError:
            pass
        log_console("warn", "discarded failed attempt", run_id=run_id)
        return None, None, None

    return gen_text, str(csd_path), str(wav_path)


def generate_until_audio_success(seed: int = None, checkpoint: Path | None = None):
    """Watchdog: retry until audio renders successfully. Only returns when both CSD and WAV exist."""
    base_seed = seed if seed is not None else int(datetime.now().timestamp() * 1000) % (2**32)
    log_console(
        "sys",
        f"generate start · base_seed={base_seed} · max_attempts={MAX_GENERATE_ATTEMPTS}",
    )
    for attempt in range(1, MAX_GENERATE_ATTEMPTS + 1):
        try_seed = base_seed + attempt * 12345
        log_console("info", f"attempt {attempt}/{MAX_GENERATE_ATTEMPTS} · seed={try_seed}")
        csd_text, csd_path, wav_path = generate_one_sample(seed=try_seed, checkpoint=checkpoint)
        if wav_path is not None:
            log_console(
                "ok",
                f"generate done · attempt {attempt} · run {Path(csd_path).parent.name}",
            )
            return csd_text, csd_path, wav_path
    log_console("err", f"generate failed after {MAX_GENERATE_ATTEMPTS} attempts")
    return None, None, None


@app.route("/")
def index():
    # The UI is served by Next.js. This route just confirms the sidecar is alive
    # when someone hits 127.0.0.1:5000 directly.
    return jsonify({
        "service": "cstore-backend",
        "ui": "http://localhost:3000",
        "endpoints": [
            "POST /api/generate",
            "GET  /api/list",
            "GET  /api/console",
            "GET  /api/models",
            "GET  /api/model",
            "POST /api/model",
            "GET  /api/qwen/status",
            "GET  /api/pollinations/status",
            "POST /api/edit",
            "POST /api/render",
            "POST /api/favorite",
            "GET  /generated/<run_id>/output.(csd|wav)",
        ],
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate CSD + WAV. Watchdog: only returns when audio renders successfully.

    Accepts optional body fields:
      - seed: int
      - checkpoint: str — relative (e.g. "Cstore_V1.0.1/best") or absolute path.
        When supplied, the active model is switched to this checkpoint first.
    """
    try:
        data = request.get_json() or {}
        seed = data.get("seed")
        if seed is not None:
            seed = int(seed)

        checkpoint = None
        ckpt_raw = data.get("checkpoint")
        if ckpt_raw:
            checkpoint = resolve_checkpoint(str(ckpt_raw))

        csd_text, csd_path, wav_path = generate_until_audio_success(
            seed=seed, checkpoint=checkpoint
        )

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
            "checkpoint": str(_active_checkpoint),
        }
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        err = str(e)
        if not err or len(err) > 200:
            err = "Generation failed"
        return jsonify({"error": err}), 500


@app.route("/api/models", methods=["GET"])
def api_models():
    """List discoverable checkpoints under model/checkpoints/ and the active one."""
    return jsonify({
        "root": str(MODEL_ROOT),
        "active": str(_active_checkpoint),
        "models": discover_checkpoints(),
    })


@app.route("/api/model", methods=["GET", "POST"])
def api_model():
    """Get or set the active checkpoint.

    POST body: { "path": "<relative or absolute path to a HF model folder>" }
    """
    global _active_checkpoint
    if request.method == "GET":
        return jsonify({"active": str(_active_checkpoint)})
    try:
        data = request.get_json() or {}
        raw = data.get("path")
        if not raw:
            return jsonify({"error": "Missing 'path' in request body."}), 400
        resolved = resolve_checkpoint(str(raw))
        # Eagerly swap so the response reflects a successful load, not just a path write.
        load_model(resolved)
        return jsonify({"active": str(_active_checkpoint), "ok": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        err = str(e) or "Failed to load checkpoint"
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


@app.route("/api/console")
def api_console():
    """Return console events newer than `since` (defaults to all buffered).

    Clients poll this endpoint and remember the returned `seq` so the next
    call only receives new lines. `limit` caps the response size.
    """
    try:
        since = int(request.args.get("since", 0) or 0)
    except ValueError:
        since = 0
    try:
        limit = max(1, min(1000, int(request.args.get("limit", 500) or 500)))
    except ValueError:
        limit = 500
    return jsonify(console_snapshot(since=since, limit=limit))


@app.route("/api/list")
def api_list():
    """List all generated runs.

    Caps at the 50 most recent for the main scroll, but always includes
    every run the user has starred as a favourite — even if it has since
    been pushed past the cap by newer generations. That way the
    Favourites drawer in the UI is guaranteed to stay populated.
    """
    runs = []
    try:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        for d in sorted(GENERATED_DIR.iterdir(), reverse=True):
            if d.is_dir():
                csd = d / "output.csd"
                wav = d / "output.wav"
                meta_path = d / "meta.json"
                entry = {
                    "run_id": d.name,
                    "has_csd": csd.exists(),
                    "has_wav": wav.exists(),
                }
                if meta_path.exists():
                    try:
                        entry["meta"] = json.loads(meta_path.read_text())
                    except (OSError, ValueError):
                        pass
                runs.append(entry)
    except (FileNotFoundError, OSError):
        runs = []

    recent = runs[:50]
    seen_ids = {r["run_id"] for r in recent}
    favourites_tail = [
        r for r in runs[50:]
        if isinstance(r.get("meta"), dict) and r["meta"].get("favorite")
        and r["run_id"] not in seen_ids
    ]
    return jsonify({"runs": recent + favourites_tail})


# ============================================================================
# External LLM editing
# ----------------------------------------------------------------------------
# Users can send a previously generated .csd to an external frontier model
# (OpenAI / Anthropic / Gemini) along with a plain-language instruction
# ("add a reverb", "transpose down an octave"). The LLM's response is
# extracted, saved as a new run, and rendered with Csound so the user can
# listen to the edit directly in the console.
#
# The API key for each provider is stored locally at ~/.cstore/keys.json
# with chmod 0600, never returned to the frontend in full (only masked
# first-4/last-4), and never logged. The key is only used server-side when
# making the outbound HTTPS call.
# ============================================================================

KEY_STORE_DIR = Path.home() / ".cstore"
KEY_STORE = KEY_STORE_DIR / "keys.json"
# Providers that require a user-supplied API key.
KEY_PROVIDERS = ("openai", "anthropic", "gemini")
# All providers the /api/edit endpoint will accept. `qwen` is served by a
# local Ollama instance (http://127.0.0.1:11434 by default) so it needs no
# key — users just run `ollama pull qwen2.5-coder:7b` once. `pollinations`
# is a free public endpoint at text.pollinations.ai: anonymous tier, rate-
# limited to ~1 req / 15 s, currently serving OpenAI's open-weight
# GPT-OSS-20B model. Verified keyless as of 2026-04 (see API docs at
# https://github.com/pollinations/pollinations/blob/master/APIDOCS.md).
SUPPORTED_PROVIDERS = KEY_PROVIDERS + ("qwen", "pollinations")
OLLAMA_BASE_URL = os.environ.get(
    "CSTORE_OLLAMA_URL", "http://127.0.0.1:11434"
).rstrip("/")
# Public endpoint that exposes the OpenAI-compatible /openai route without
# authentication. Overridable via env var so a self-hosted mirror can be
# pointed at instead.
POLLINATIONS_BASE_URL = os.environ.get(
    "CSTORE_POLLINATIONS_URL", "https://text.pollinations.ai"
).rstrip("/")

LLM_SYSTEM_PROMPT = """You are an expert Csound programmer working on .csd files that will be rendered immediately with `csound -W -d -m135 -o output.wav input.csd`. Your output is fed *directly* to the csound binary — there is no human review step.

Task format
-----------
You will be given:
  1. A complete .csd file (from <CsoundSynthesizer> to </CsoundSynthesizer>).
  2. A short natural-language instruction describing a modification.

Your job is to produce a *new, complete* .csd file that applies the instruction while preserving the original musical intent. Think of it as an in-place edit, not a rewrite.

CRITICAL: TARGET CSOUND 6 SYNTAX ONLY
-------------------------------------
The .csd will be compiled by Csound 6.18 (November 2022). Csound 7 features are NOT available and will cause a parser failure. Specifically:

DO use traditional opcode syntax with the result variables on the LEFT and arguments separated by spaces/tabs (NO parentheses, NO `=`):

    aL, aR  reverbsc  asigL, asigR, 0.85, 8000
    asig    poscil    0.5, 440, 1
    aout    oscil     0.3, 220
    kenv    linen     1, 0.1, p3, 0.2
    aL, aR  pan2      amono, 0.5

DO NOT use Csound 7 function-call syntax. These WILL fail:

    aL, aR = reverbsc(asigL, asigR, 0.85, 8000)        ; ← WRONG
    asig   = poscil(0.5, 440, 1)                       ; ← WRONG
    a1     = a1 * aenv                                 ; ← OK only for simple arithmetic, NOT for opcodes

For arithmetic on a-rate / k-rate signals, `=` IS allowed (a1 = a1 * 1.2 is fine).
For calling an opcode, ALWAYS use the traditional form.

Other Csound-7-only features to avoid: bracket array literals `[1, 2, 3]`, `#include` you didn't see in the original, `if/then/endif` one-liners not present before, `chn_k` replaced with `chnget`.

Exact signatures of effect opcodes (get these EXACTLY right — wrong arity = parse error):

    aL, aR     reverbsc   ainL, ainR, kFB, kFCO[, iSR, iPM]      ; 2 a-rate ins, NOT 1
    aout       reverb     ainput, kvtime
    aout       freeverb   ainL, ainR, ksize, kdamp               ; returns MONO mix
    aout       comb       ainput, krvt, ilpt
    aout       vcomb      ainput, krvt, xlpt, imaxlpt
    aout       butterlp   ainput, kfreq
    aout       butterhp   ainput, kfreq
    aout       butterbp   ainput, kfreq, kbw
    aout       moogladder ainput, kcf, kres
    aout       delay      ainput, idltime
    aout       delayr     idltime
               delayw     ainput
    aL, aR     pan2       amono, kpos                            ; kpos 0..1
    aL, aR     hrtfmove2  ainput, kAZ, kEL, Sleft, Sright
    kenv       linen      kamp, irise, idur, idec
    kenv       linseg     ia, idur1, ib[, idur2, ic...]
    kenv       expseg     ia, idur1, ib[, idur2, ic...]
    aenv       adsr       iatt, idec, islev, irel

When the instruction says "add reverb 30% wet on a mono signal", a reliable Csound 6 pattern is:

    ; before: outs asig, asig
    aL, aR   reverbsc   asig, asig, 0.85, 8000
    outs     asig + 0.3 * aL, asig + 0.3 * aR

Hard rules (violations cause render failure)
--------------------------------------------
1. Output ONLY the .csd text — from the literal string `<CsoundSynthesizer>` to the literal string `</CsoundSynthesizer>` inclusive. No markdown fences, no prose, no explanations, no "Here is the modified file:" preamble.
2. The file MUST contain, in order: `<CsoundSynthesizer>`, `<CsOptions>...</CsOptions>` (may be empty), `<CsInstruments>...</CsInstruments>`, `<CsScore>...</CsScore>`, `</CsoundSynthesizer>`.
3. Every `instr N` MUST be closed by `endin` on its own line. Every `opcode name,...` MUST be closed by `endop`.
4. The score MUST end with `e` on its own line before `</CsScore>`.
5. `sr`, `kr`/`ksmps`, `nchnls`, and `0dbfs` in the header must remain consistent with the original unless the instruction explicitly changes them.
6. Keep `ftgen` / `f`-statement indices stable — if instr 1 uses `1`, don't silently renumber to `f 99`. Only add new f-statements with fresh indices.
7. Output format: one statement per line, tabs or spaces OK, but do NOT put `endin` on the same line as code.
8. Match the syntax style already present in the file. If the original uses `outs a1, a1` and `a1 poscil p4, p5, 1`, keep that style — don't switch to a different convention.
9. If you cannot apply the instruction without breaking the file, output the ORIGINAL .csd unchanged rather than producing broken code.

Editing principles
------------------
- Make the *smallest* change that satisfies the instruction. Do not refactor unrelated code.
- When asked for audio effects (reverb, delay, chorus, filter), insert them BEFORE `outs`/`out`, operating on the same variables that were being sent to output.
- When asked to change dynamics/amplitude, prefer editing existing `kenv`/`aenv`/`p4` usage over introducing new signal paths.
- When asked to transpose, edit the frequency arguments in the score (`p5` or explicit `cpspch` calls) or the oscillator's frequency input — do not change `sr`.
- Keep original comments; add a single concise comment above your change if helpful.

Reasoning style
---------------
Think step by step in your internal reasoning:
  1. Parse the original .csd: identify orchestra vs. score, list the instruments and what opcodes they use.
  2. Locate the minimal spans the instruction targets.
  3. Draft the replacement.
  4. Mentally "render" it — check every `instr` has an `endin`, the score ends with `e`, and the header is intact.
  5. Emit only the final .csd.

Remember: the file you emit will be compiled by `csound` immediately. Any extra character outside the <CsoundSynthesizer>...</CsoundSynthesizer> block is a syntax error."""


def _load_keys() -> dict:
    if not KEY_STORE.exists():
        return {}
    try:
        data = json.loads(KEY_STORE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_keys(keys: dict) -> None:
    KEY_STORE_DIR.mkdir(parents=True, exist_ok=True)
    KEY_STORE.write_text(json.dumps(keys, indent=2), encoding="utf-8")
    # Best-effort: restrict to owner read/write only.
    try:
        os.chmod(KEY_STORE, 0o600)
    except OSError:
        pass


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) < 10:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"


@app.route("/api/keys", methods=["GET"])
def api_keys_get():
    """Return which key-providers have a key saved, masked for display only.

    `qwen` runs on local Ollama and is intentionally excluded here — it has
    no key to store. The frontend queries /api/qwen/status instead.
    """
    keys = _load_keys()
    return jsonify({
        p: {
            "present": p in keys and bool(keys[p]),
            "masked": _mask_key(keys.get(p, "")),
        }
        for p in KEY_PROVIDERS
    })


@app.route("/api/keys", methods=["POST"])
def api_keys_post():
    """Store a provider's API key. Body: {provider, key}."""
    data = request.get_json() or {}
    provider = data.get("provider")
    key = (data.get("key") or "").strip()
    if provider not in KEY_PROVIDERS:
        return jsonify({"error": f"Provider '{provider}' does not use a key"}), 400
    if not key:
        return jsonify({"error": "Missing 'key'"}), 400
    keys = _load_keys()
    keys[provider] = key
    _save_keys(keys)
    return jsonify({"ok": True, "provider": provider, "masked": _mask_key(key)})


@app.route("/api/keys", methods=["DELETE"])
def api_keys_delete():
    """Remove a stored key. Body: {provider}."""
    data = request.get_json() or {}
    provider = data.get("provider")
    if provider not in KEY_PROVIDERS:
        return jsonify({"error": f"Provider '{provider}' does not use a key"}), 400
    keys = _load_keys()
    if provider in keys:
        del keys[provider]
        _save_keys(keys)
    return jsonify({"ok": True})


def _extract_csd(text: str) -> str | None:
    """Pull a valid <CsoundSynthesizer>...</CsoundSynthesizer> block out of an
    LLM response, handling markdown fences and surrounding chatter."""
    if not text:
        return None
    # Strip markdown fences if present.
    fence = re.search(r"```(?:csd|csound|xml)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    block = re.search(
        r"<CsoundSynthesizer>.*?</CsoundSynthesizer>", text, re.DOTALL
    )
    if not block:
        return None
    return block.group(0).strip() + "\n"


def _call_openai(key: str, model: str, system: str, user: str, timeout: int = 90) -> str:
    import requests
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _call_anthropic(key: str, model: str, system: str, user: str, timeout: int = 90) -> str:
    import requests
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Anthropic {r.status_code}: {r.text[:300]}")
    data = r.json()
    return "".join(
        block.get("text", "") for block in data.get("content", [])
        if block.get("type") == "text"
    )


def _call_gemini(key: str, model: str, system: str, user: str, timeout: int = 90) -> str:
    import requests
    # Gemini's REST API takes the system instructions as a top-level field.
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _call_qwen(
    _key: str | None,
    model: str,
    system: str,
    user: str,
    timeout: int = 900,
    think: bool = True,
) -> str:
    """Call a Qwen model through a local Ollama server.

    Ollama is a free local LLM runner (https://ollama.com). If it's running,
    POST /api/chat speaks a simple JSON protocol with no authentication. The
    user pulls a model once (e.g. `ollama pull qwen3.6:35b-a3b-coding-mxfp8`)
    and it stays available offline. The `_key` argument is ignored and kept
    only to match the dispatch signature used by the cloud providers.

    Quality-tuned for code editing:

    - `think` turns ON Qwen 3/3.6's chain-of-thought. This is the single
      biggest quality lever for a reasoning model on a specialised task
      like Csound editing. With it on, latency is ~1-3 min per edit; off,
      it drops to ~15-25 s but quality drops noticeably on anything more
      involved than trivial edits. Exposed as a kwarg so the /api/edit
      caller can flip it from the UI. Non-reasoning models (qwen2.5,
      qwen2.5-coder) ignore the flag.
    - Sampling params follow Qwen's own published recommendations for
      coding tasks (see the HF model card): low temperature, tight top_p,
      small top_k. Reduces "creative" mutations that break Csound syntax.
    - num_ctx: 32 K — plenty for a .csd + instruction + the thinking
      trace + replacement .csd. Qwen 3.6 supports 256 K but that would
      allocate way more KV cache than we need.
    """
    import requests
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # Toggle reasoning. Default on because the quality of the
                # final .csd depends strongly on letting the model plan
                # the edit first; the UI can pass `think: false` when the
                # user wants a fast, cheap draft.
                "think": bool(think),
                "options": {
                    # Conservative sampling — we want a near-deterministic
                    # edit, not a creative rewrite. Values from Qwen team's
                    # own recommended defaults for code.
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 20,
                    "min_p": 0.0,
                    "repeat_penalty": 1.05,
                    "num_ctx": 32768,
                    # Enough room for the model to think AND emit a full
                    # replacement .csd. Typical thinking trace is 1-3 k
                    # tokens; a .csd is rarely over 1.5 k tokens.
                    "num_predict": 8192,
                },
            },
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. "
            "Install Ollama from https://ollama.com and run `ollama serve`."
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s — is the model still loading?"
        ) from e

    if r.status_code == 404:
        # Ollama returns 404 when the requested model isn't pulled locally.
        raise RuntimeError(
            f"Model '{model}' is not installed in Ollama. "
            f"Run `ollama pull {model}` and try again."
        )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
    data = r.json()
    msg_obj = data.get("message") or {}
    msg = msg_obj.get("content") or ""
    # Defensive fallback: if some future Ollama version still emits the
    # answer inside `thinking` even with think=false, accept it rather
    # than raising — _extract_csd downstream strips non-.csd chatter.
    if not msg:
        msg = msg_obj.get("thinking") or ""
    if not msg:
        raise RuntimeError("Ollama returned an empty response")
    return msg


def _call_pollinations(
    _key: str | None,
    model: str,
    system: str,
    user: str,
    timeout: int = 120,
) -> str:
    """Call the public Pollinations text API — no API key required.

    Pollinations (https://pollinations.ai) runs a free, anonymous-access
    text gateway at `text.pollinations.ai/openai` that speaks the OpenAI
    Chat Completions protocol. The `anonymous` tier is rate-limited to
    roughly one request every 15 seconds per IP; there is no signup, no
    token, and no credit card. The only model available at that tier at
    the time of writing is `openai` (alias for `openai-fast`), which
    routes to OpenAI's open-weight **GPT-OSS-20B** served on OVH.

    Reference: https://github.com/pollinations/pollinations/blob/master/APIDOCS.md

    The `_key` argument is ignored; it exists only so this function shares
    the (_key, model, system, user) signature used by the cloud providers
    in _LLM_DISPATCH.
    """
    import requests
    try:
        r = requests.post(
            f"{POLLINATIONS_BASE_URL}/openai",
            headers={"Content-Type": "application/json"},
            json={
                "model": model or "openai",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Low temperature because Csound syntax is brittle — the
                # model benefits more from "careful editing" than from
                # creative rewrites, same as the other providers here.
                "temperature": 0.3,
            },
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach Pollinations at {POLLINATIONS_BASE_URL}. "
            "Check your internet connection."
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"Pollinations timed out after {timeout}s — the free tier can be slow "
            "under load. Try again in a moment."
        ) from e

    if r.status_code == 429:
        raise RuntimeError(
            "Pollinations rate limit hit — the anonymous tier allows about one "
            "request every 15 seconds. Wait a moment and retry."
        )
    if r.status_code != 200:
        raise RuntimeError(f"Pollinations {r.status_code}: {r.text[:300]}")
    try:
        data = r.json()
    except ValueError as e:
        raise RuntimeError("Pollinations returned non-JSON response") from e
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Pollinations returned no choices")
    msg = choices[0].get("message", {}) or {}
    content = msg.get("content") or msg.get("reasoning_content") or ""
    if not content:
        raise RuntimeError("Pollinations returned an empty response")
    return content


_LLM_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "qwen": _call_qwen,
    "pollinations": _call_pollinations,
}


@app.route("/api/favorite", methods=["POST"])
def api_favorite():
    """Toggle or set a run's `favorite` flag in its meta.json.

    Body:
      {
        "run_id":   "YYYYMMDD_HHMMSS_xxxxxxxx",
        "favorite": true | false      # optional — omit to toggle current value
      }

    Persists as a single boolean field inside the run's `meta.json`, creating
    that file if it doesn't exist yet (generations don't ship with one).
    Returns the run's full meta so the caller can update its UI without
    making a second request.
    """
    try:
        data = request.get_json() or {}
        run_id = data.get("run_id")
        if not run_id or ".." in run_id or "/" in run_id or "\\" in run_id:
            return jsonify({"error": "Invalid run_id"}), 400

        run_dir = GENERATED_DIR / run_id
        if not run_dir.is_dir():
            return jsonify({"error": f"Run not found: {run_id}"}), 404

        meta_path = run_dir / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except (OSError, ValueError):
                # Corrupted / unreadable — start fresh but don't clobber
                # silently: log so the user can investigate if it matters.
                log_console(
                    "warn",
                    f"could not parse existing meta.json · rewriting",
                    run_id=run_id,
                )

        raw = data.get("favorite")
        if raw is None:
            favorite = not bool(meta.get("favorite", False))
        else:
            favorite = bool(raw)

        meta["favorite"] = favorite
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        log_console(
            "info",
            f"favorite · {'on' if favorite else 'off'}",
            run_id=run_id,
        )
        return jsonify({"ok": True, "run_id": run_id, "meta": meta})
    except Exception as e:
        err = str(e) or "Favorite toggle failed"
        return jsonify({"error": err[:300]}), 500


@app.route("/api/render", methods=["POST"])
def api_render():
    """Render a user-edited .csd as a new run.

    Body:
      {
        "csd":          "<CsoundSynthesizer>…</CsoundSynthesizer>",
        "derived_from": "<optional source run_id>"
      }

    Unlike /api/edit this path never calls any LLM — the .csd text is written
    verbatim, then handed to Csound. Used by the UI's manual-edit window so
    users can tweak the model's output by hand and re-render.
    """
    try:
        data = request.get_json() or {}
        csd_raw = data.get("csd")
        derived_from = (data.get("derived_from") or "").strip() or None

        if not isinstance(csd_raw, str) or not csd_raw.strip():
            return jsonify({"error": "Missing 'csd' text"}), 400
        if "<CsoundSynthesizer>" not in csd_raw or "</CsoundSynthesizer>" not in csd_raw:
            return jsonify({
                "error": "CSD must contain <CsoundSynthesizer> and </CsoundSynthesizer>"
            }), 400
        if derived_from and (
            ".." in derived_from or "/" in derived_from or "\\" in derived_from
        ):
            return jsonify({"error": "Invalid derived_from"}), 400

        # Ensure the text ends with a newline so Csound's lexer is happy.
        csd_text = csd_raw if csd_raw.endswith("\n") else csd_raw + "\n"

        new_run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_manual_" + uuid.uuid4().hex[:8]
        )
        run_dir = GENERATED_DIR / new_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        csd_path = run_dir / "output.csd"
        wav_path = run_dir / "output.wav"
        meta_path = run_dir / "meta.json"

        csd_path.write_text(csd_text, encoding="utf-8")
        log_console(
            "sys",
            f"manual render · {len(csd_text)} chars · "
            f"{('from ' + derived_from) if derived_from else 'no source'}",
            run_id=new_run_id,
        )

        meta = {
            "kind": "manual_edit",
            "derived_from": derived_from,
        }

        if not render_csd_to_wav(csd_path, wav_path):
            meta["render_ok"] = False
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return jsonify({
                "error": (
                    "Csound could not render audio from the supplied .csd. "
                    "Source is saved for inspection."
                ),
                "run_id": new_run_id,
                "csd_url": f"/generated/{new_run_id}/output.csd",
            }), 422

        meta["render_ok"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return jsonify({
            "success": True,
            "csd": csd_text,
            "run_id": new_run_id,
            "csd_url": f"/generated/{new_run_id}/output.csd",
            "wav_url": f"/generated/{new_run_id}/output.wav",
            "derived_from": derived_from,
        })
    except Exception as e:
        err = str(e) or "Render failed"
        return jsonify({"error": err[:300]}), 500


@app.route("/api/qwen/status", methods=["GET"])
def api_qwen_status():
    """Probe the local Ollama server and list installed qwen models.

    The frontend uses this to render an honest status line ("Ollama running ·
    3 qwen models installed" vs. "Ollama not reachable — install from …").
    Never throws: any failure surfaces as `{available: false, error: "…"}`.
    """
    import requests
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if r.status_code != 200:
            return jsonify({
                "available": False,
                "base_url": OLLAMA_BASE_URL,
                "error": f"Ollama returned HTTP {r.status_code}",
                "models": [],
            })
        raw = r.json().get("models") or []
        models = [
            {"name": m.get("name", ""), "size": m.get("size", 0)}
            for m in raw
            if isinstance(m, dict) and m.get("name")
        ]
        qwen_models = [m for m in models if "qwen" in m["name"].lower()]
        return jsonify({
            "available": True,
            "base_url": OLLAMA_BASE_URL,
            "models": models,
            "qwen_models": qwen_models,
        })
    except Exception as e:
        return jsonify({
            "available": False,
            "base_url": OLLAMA_BASE_URL,
            "error": str(e)[:200],
            "models": [],
        })


@app.route("/api/pollinations/status", methods=["GET"])
def api_pollinations_status():
    """Probe the public Pollinations text endpoint.

    Returns availability + the subset of models listed for the `anonymous`
    tier so the UI can populate the model dropdown honestly with whatever
    is actually callable without a key. Never throws: any failure surfaces
    as `{available: false, error: "…"}`.
    """
    import requests
    try:
        # /models is keyless per the API docs (model listing never requires auth).
        r = requests.get(f"{POLLINATIONS_BASE_URL}/models", timeout=3)
        if r.status_code != 200:
            return jsonify({
                "available": False,
                "base_url": POLLINATIONS_BASE_URL,
                "error": f"Pollinations returned HTTP {r.status_code}",
                "models": [],
            })
        raw = r.json()
        if not isinstance(raw, list):
            raw = []
        # Keep only models callable from the anonymous tier — those are the
        # only ones the user can actually invoke here without signing up.
        anon_models: list[dict] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            tier = (m.get("tier") or "").lower()
            if tier and tier != "anonymous":
                continue
            name = m.get("name") or ""
            if not name:
                continue
            anon_models.append({
                "name": name,
                "description": m.get("description", ""),
                "reasoning": bool(m.get("reasoning", False)),
                "aliases": m.get("aliases") or [],
            })
        return jsonify({
            "available": True,
            "base_url": POLLINATIONS_BASE_URL,
            "models": anon_models,
            "rate_limit_seconds": 15,  # documented anonymous-tier limit
        })
    except Exception as e:
        return jsonify({
            "available": False,
            "base_url": POLLINATIONS_BASE_URL,
            "error": str(e)[:200],
            "models": [],
        })


@app.route("/api/edit", methods=["POST"])
def api_edit():
    """Edit an existing run with an external LLM.

    Body:
      {
        "run_id":      "...",
        "provider":    "openai" | "anthropic" | "gemini" | "qwen" | "pollinations",
        "model":       e.g. "gpt-5.4" · "claude-opus-4-7" · "gemini-3.1-pro"
                           · "qwen3.6:35b-a3b-coding-mxfp8" · "openai"
                             (pollinations alias → GPT-OSS-20B),
        "instruction": "Add a reverb, keep the melody."
      }
    """
    try:
        data = request.get_json() or {}
        run_id = data.get("run_id")
        provider = data.get("provider")
        model = (data.get("model") or "").strip()
        instruction = (data.get("instruction") or "").strip()
        # Deep reasoning toggle — only meaningful for `qwen` (local Ollama).
        # Accept it from the body; default on to preserve previous behaviour.
        # Anything that isn't an explicit `false` stays on.
        raw_think = data.get("think", True)
        think = False if raw_think is False else True

        if provider not in SUPPORTED_PROVIDERS:
            return jsonify({"error": f"Unsupported provider: {provider}"}), 400
        if not model:
            return jsonify({"error": "Missing model name"}), 400
        if not instruction:
            return jsonify({"error": "Missing instruction"}), 400
        if not run_id or ".." in run_id or "/" in run_id or "\\" in run_id:
            return jsonify({"error": "Invalid run_id"}), 400

        source_csd = GENERATED_DIR / run_id / "output.csd"
        if not source_csd.exists():
            return jsonify({"error": f"Source run not found: {run_id}"}), 404

        # Qwen runs locally via Ollama and therefore has no API key to store.
        # For every other provider we require a pre-saved key.
        key: str | None = None
        if provider in KEY_PROVIDERS:
            keys = _load_keys()
            key = keys.get(provider)
            if not key:
                return jsonify({
                    "error": f"No API key stored for {provider}. Save one via /api/keys first.",
                }), 401

        original = source_csd.read_text(encoding="utf-8")
        user_msg = (
            f"User instruction:\n{instruction}\n\n"
            f"Current .csd:\n{original}"
        )

        reasoning_tag = (
            f" · think={'on' if think else 'off'}" if provider == "qwen" else ""
        )
        log_console(
            "sys",
            f"edit · {provider}/{model}{reasoning_tag} · run {run_id} · “{instruction[:80]}”",
        )
        t_llm = time.time()
        try:
            # Only qwen (Ollama) understands `think`; cloud providers don't,
            # and passing unknown kwargs into their helpers would raise a
            # TypeError.
            if provider == "qwen":
                raw = _call_qwen(
                    key, model, LLM_SYSTEM_PROMPT, user_msg, think=think
                )
            else:
                raw = _LLM_DISPATCH[provider](key, model, LLM_SYSTEM_PROMPT, user_msg)
        except Exception as e:
            # Anything from timeout, network error, HTTP error, or JSON shape.
            log_console("err", f"LLM call failed: {str(e)[:200]}")
            return jsonify({"error": f"LLM call failed: {str(e)[:300]}"}), 502
        log_console(
            "info",
            f"LLM responded · {len(raw or '')} chars in {time.time() - t_llm:.2f}s",
        )

        new_csd = _extract_csd(raw)
        if not new_csd:
            return jsonify({
                "error": (
                    "LLM response did not contain a valid "
                    "<CsoundSynthesizer>…</CsoundSynthesizer> block."
                ),
            }), 502

        new_run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_edit_" + uuid.uuid4().hex[:8]
        )
        run_dir = GENERATED_DIR / new_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        csd_path = run_dir / "output.csd"
        wav_path = run_dir / "output.wav"
        meta_path = run_dir / "meta.json"

        csd_path.write_text(new_csd, encoding="utf-8")

        meta = {
            "kind": "edit",
            "derived_from": run_id,
            "provider": provider,
            "model": model,
            "instruction": instruction,
        }

        if not render_csd_to_wav(csd_path, wav_path):
            # Keep the csd so the user can inspect the broken edit.
            meta["render_ok"] = False
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return jsonify({
                "error": (
                    "LLM produced a .csd but Csound could not render audible "
                    "audio from it. Source is saved for inspection."
                ),
                "run_id": new_run_id,
                "csd_url": f"/generated/{new_run_id}/output.csd",
            }), 422

        meta["render_ok"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return jsonify({
            "success": True,
            "csd": new_csd,
            "run_id": new_run_id,
            "csd_url": f"/generated/{new_run_id}/output.csd",
            "wav_url": f"/generated/{new_run_id}/output.wav",
            "derived_from": run_id,
            "provider": provider,
            "model": model,
            "instruction": instruction,
        })
    except Exception as e:
        err = str(e) or "Edit failed"
        return jsonify({"error": err[:300]}), 500


if __name__ == "__main__":
    banner = [
        "CStore backend (Python sidecar)",
        "  API          : http://127.0.0.1:5000",
        "  UI (Next.js) : http://localhost:3000",
        f"  Default ckpt : {DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)}",
        f"  Outputs to   : {GENERATED_DIR.relative_to(PROJECT_ROOT)}",
        "Switch checkpoints at runtime via the UI selector or POST /api/model.",
        "Csound (csound CLI) required for audio rendering.",
    ]
    for line in banner:
        print(line)
        log_console("sys", line)
    # Probe csound once so the terminal reports its version at boot.
    try:
        r = subprocess.run(
            ["csound", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        version = ((r.stderr or r.stdout or "").strip().splitlines() or ["(no output)"])[0]
        log_console("sys", f"csound detected · {version}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log_console("warn", f"csound probe failed · {e}")
    app.run(host="127.0.0.1", port=5000, debug=False)
