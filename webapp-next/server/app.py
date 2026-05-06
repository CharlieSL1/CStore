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
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
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
PROMPT = "<CsoundSynthesizer>"
DEFAULT_MAX_NEW_TOKENS = 400
MIN_NEW_TOKENS = 100
MIN_REQUESTED_MAX_TOKENS = 100
MAX_REQUESTED_MAX_TOKENS = 500
STARTER_DURATION_SEC = 1.5
MIN_NOTE_DURATION_SEC = 0.1
MAX_NOTE_DURATION_SEC = 10.0
MIN_NOTE_MIDI = 24
MAX_NOTE_MIDI = 108
DEFAULT_STARTER_COUNT = 6
MAX_STARTER_COUNT = 12
MAX_LLM_CHILD_ATTEMPTS = 3
NOTE_MODES = ("single", "arpeggio", "chord")
SAFE_PEAK_TARGET = 0.82
SAFE_RMS_TARGET = 0.25


def llm_var_profile(variation_temp: float) -> tuple[str, str, float]:
    """Map slider [0..1] to (tier, prompt mode, provider temperature)."""
    if variation_temp <= 0.33:
        return ("low", "subtle", 0.25)
    if variation_temp <= 0.66:
        return ("medium", "moderate", 0.5)
    return ("high", "aggressive-style-switch", 0.8)


def _mix_seed32(value: int) -> int:
    """Mix an integer into a well-distributed 32-bit seed."""
    x = value & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def _auto_seed32() -> int:
    """Best-effort high-entropy seed for runs where user didn't pin one."""
    entropy = (
        time.time_ns()
        ^ random.SystemRandom().getrandbits(32)
        ^ uuid.uuid4().int
        ^ os.getpid()
    )
    return _mix_seed32(entropy)


def _derive_child_seed(base_seed: int, child_index: int) -> int:
    """Deterministically derive one child seed from a base batch seed."""
    return _mix_seed32(base_seed + (child_index + 1) * 0x9E3779B9)


def _derive_attempt_seed(base_seed: int, attempt_index: int, child_index: int = 0) -> int:
    """Deterministically derive one retry seed per attempt and child index."""
    return _mix_seed32(
        base_seed + (attempt_index + 1) * 0x85EBCA6B + child_index * 0xC2B2AE35
    )

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


def _valid_run_id(run_id: str | None) -> bool:
    return bool(run_id) and ".." not in run_id and "/" not in run_id and "\\" not in run_id


def _read_run_meta(run_id: str) -> dict:
    meta_path = GENERATED_DIR / run_id / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        log_console("warn", "could not parse existing meta.json", run_id=run_id)
        return {}


def _write_run_meta(run_id: str, meta: dict) -> None:
    meta_path = GENERATED_DIR / run_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _merge_run_meta(run_id: str, patch: dict) -> dict:
    meta = _read_run_meta(run_id)
    meta.update(patch)
    _write_run_meta(run_id, meta)
    return meta


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


def validate_max_tokens(raw: object, default: int = DEFAULT_MAX_NEW_TOKENS) -> int:
    """Validate the user-facing generation budget before model-level clamping."""
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < MIN_REQUESTED_MAX_TOKENS or value > MAX_REQUESTED_MAX_TOKENS:
        raise ValueError(
            f"max_tokens must be between {MIN_REQUESTED_MAX_TOKENS} and {MAX_REQUESTED_MAX_TOKENS}"
        )
    return value


def effective_generation_budget(model, prompt_token_count: int, requested_max_tokens: int) -> tuple[int, int]:
    """Clamp requested output tokens to the model's total context window."""
    context_limit = int(
        getattr(model.config, "n_positions", None)
        or getattr(model.config, "n_ctx", None)
        or getattr(model.config, "max_position_embeddings", None)
        or 512
    )
    available = max(1, context_limit - prompt_token_count)
    return min(requested_max_tokens, available), context_limit


def parse_i_statement(line: str) -> dict | None:
    """Parse simple Csound i-statements while preserving p-fields for rewriting."""
    stripped = line.strip()
    m = re.match(r"^i\s*(\S+)\s+(.*)$", stripped, re.I)
    if not m:
        return None
    fields = m.group(2).split()
    if len(fields) < 2:
        return None
    return {"instr": m.group(1), "fields": fields}


def format_i_statement(
    parsed: dict,
    start: float,
    dur: float,
    freq_ratio: float | None = None,
    freq_override: float | None = None,
) -> str:
    fields = list(parsed["fields"])
    fields[0] = f"{start:.3f}".rstrip("0").rstrip(".")
    fields[1] = f"{dur:.3f}".rstrip("0").rstrip(".")
    if len(fields) < 3:
        fields.append("220")
    if len(fields) >= 3:
        if freq_override is not None:
            fields[2] = f"{freq_override:.3f}".rstrip("0").rstrip(".")
        elif freq_ratio is not None:
            try:
                freq = float(fields[2])
                if 20 <= freq <= 5000:
                    fields[2] = f"{freq * freq_ratio:.3f}".rstrip("0").rstrip(".")
            except ValueError:
                pass
    return f"i {parsed['instr']} {' '.join(fields)}"


def validate_note_duration(raw: object, default: float = STARTER_DURATION_SEC) -> float:
    if raw is None or raw == "":
        return default
    value = float(raw)
    if value < MIN_NOTE_DURATION_SEC or value > MAX_NOTE_DURATION_SEC:
        raise ValueError(
            f"note_duration must be between {MIN_NOTE_DURATION_SEC:g} and {MAX_NOTE_DURATION_SEC:g} seconds"
        )
    return value


def validate_note_mode(raw: object, default: str = "single") -> str:
    mode = (str(raw or default)).strip().lower()
    aliases = {"tone": "single", "scale": "arpeggio"}
    mode = aliases.get(mode, mode)
    if mode not in NOTE_MODES:
        raise ValueError(f"note_mode must be one of: {', '.join(NOTE_MODES)}")
    return mode


def validate_note_range_mode(raw: object, default: str = "auto") -> str:
    mode = (str(raw or default)).strip().lower()
    if mode not in ("auto", "manual"):
        raise ValueError("note_range_mode must be one of: auto, manual")
    return mode


def validate_note_midi(raw: object, label: str, default: int) -> int:
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < MIN_NOTE_MIDI or value > MAX_NOTE_MIDI:
        raise ValueError(
            f"{label} must be between {MIN_NOTE_MIDI} and {MAX_NOTE_MIDI}"
        )
    return value


def hz_to_midi(freq: float) -> int:
    if freq <= 0:
        return 57
    midi = int(round(69 + 12 * (math.log(freq / 440.0, 2))))
    return max(MIN_NOTE_MIDI, min(MAX_NOTE_MIDI, midi))


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def pick_distinct_midis(
    rng: random.Random, count: int, low_midi: int, high_midi: int
) -> list[int]:
    lo = max(MIN_NOTE_MIDI, min(MAX_NOTE_MIDI, low_midi))
    hi = max(MIN_NOTE_MIDI, min(MAX_NOTE_MIDI, high_midi))
    if lo > hi:
        lo, hi = hi, lo
    pool = list(range(lo, hi + 1))
    if len(pool) >= count:
        chosen = rng.sample(pool, count)
    else:
        chosen = [rng.choice(pool) for _ in range(count)]
    return chosen


def _min_adjacent_gap(midis: list[int]) -> int:
    if len(midis) < 2:
        return 12
    ordered = sorted(midis)
    return min(b - a for a, b in zip(ordered, ordered[1:]))


def _total_span(midis: list[int]) -> int:
    if not midis:
        return 0
    return max(midis) - min(midis)


def pick_spread_midis(
    rng: random.Random,
    count: int,
    low_midi: int,
    high_midi: int,
    min_gap: int,
    min_span: int,
) -> list[int]:
    """Pick notes that are distinct *and* musically separated.

    Falls back gracefully when the requested range is too narrow, but still
    avoids duplicated MIDI notes.
    """
    lo = max(MIN_NOTE_MIDI, min(MAX_NOTE_MIDI, low_midi))
    hi = max(MIN_NOTE_MIDI, min(MAX_NOTE_MIDI, high_midi))
    if lo > hi:
        lo, hi = hi, lo
    width = hi - lo + 1
    effective_gap = max(1, min(min_gap, width // max(1, count - 1)))
    effective_span = max(1, min(min_span, max(1, width - 1)))

    # Deterministic bounded search for a high-quality set first.
    attempts = 96
    for _ in range(attempts):
        picked = pick_distinct_midis(rng, count, lo, hi)
        if len(set(picked)) < count:
            continue
        if _min_adjacent_gap(picked) < effective_gap:
            continue
        if _total_span(picked) < effective_span:
            continue
        return picked

    # Fallback: widest unique picks in range.
    pool = list(range(lo, hi + 1))
    rng.shuffle(pool)
    picked = sorted(pool[: max(1, min(count, len(pool)))])
    if len(picked) < count:
        picked = pick_distinct_midis(rng, count, lo, hi)
        picked = sorted(list(dict.fromkeys(picked)))
        while len(picked) < count:
            picked.append(rng.randint(lo, hi))
            picked = sorted(list(dict.fromkeys(picked)))
    return picked[:count]


def format_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def apply_starter_score(
    csd_text: str,
    note_mode: str,
    duration: float = STARTER_DURATION_SEC,
    child_index: int = 0,
    note_range_mode: str = "auto",
    note_low_midi: int = 48,
    note_high_midi: int = 72,
    rng_seed: int | None = None,
) -> str:
    """Rewrite the score into a short audition: single note, arpeggio, or chord."""
    m = re.search(r"<CsScore>\s*(.*?)\s*</CsScore>", csd_text, re.DOTALL | re.I)
    if not m:
        return csd_text

    note_mode = validate_note_mode(note_mode)
    raw_lines = [line.strip() for line in m.group(1).splitlines() if line.strip()]
    f_lines = [line for line in raw_lines if re.match(r"^f\s*\d+", line, re.I)]
    first_i = next((parse_i_statement(line) for line in raw_lines if parse_i_statement(line)), None)
    if first_i is None:
        first_i = {"instr": "1", "fields": ["0", format_float(duration)]}

    try:
        base_freq = float(first_i["fields"][2]) if len(first_i["fields"]) >= 3 else 220.0
    except (ValueError, TypeError):
        base_freq = 220.0
    note_range_mode = validate_note_range_mode(note_range_mode)
    if note_range_mode == "auto":
        base_midi = hz_to_midi(base_freq)
        # Wide default spread: roughly two octaves and change.
        low_midi = max(MIN_NOTE_MIDI, base_midi - 14)
        high_midi = min(MAX_NOTE_MIDI, base_midi + 14)
    else:
        low_midi = min(note_low_midi, note_high_midi)
        high_midi = max(note_low_midi, note_high_midi)
    if low_midi == high_midi:
        high_midi = min(MAX_NOTE_MIDI, low_midi + 1)

    seed_base = rng_seed if rng_seed is not None else 0
    rng = random.Random(seed_base + child_index * 7919 + 17)
    if note_mode == "arpeggio":
        picked = pick_spread_midis(
            rng,
            count=4,
            low_midi=low_midi,
            high_midi=high_midi,
            min_gap=3,
            min_span=9,
        )
        rng.shuffle(picked)
        step = duration / len(picked)
        score_lines = [
            format_i_statement(first_i, i * step, step, freq_override=midi_to_hz(midi))
            for i, midi in enumerate(picked)
        ]
    elif note_mode == "chord":
        picked = sorted(
            pick_spread_midis(
                rng,
                count=3,
                low_midi=low_midi,
                high_midi=high_midi,
                min_gap=4,
                min_span=10,
            )
        )
        score_lines = [
            format_i_statement(first_i, 0, duration, freq_override=midi_to_hz(midi))
            for midi in picked
        ]
    else:
        picked = pick_distinct_midis(rng, 1, low_midi, high_midi)
        score_lines = [format_i_statement(first_i, 0, duration, freq_override=midi_to_hz(picked[0]))]

    new_score = "\n".join(f_lines + score_lines + ["e"])
    return csd_text[: m.start()] + f"<CsScore>\n{new_score}\n</CsScore>" + csd_text[m.end() :]


def apply_output_envelope(csd_text: str) -> str:
    """Add a short Csound 6-safe linen envelope before simple output opcodes."""
    lines = csd_text.splitlines()
    out_lines: list[str] = []
    in_instr = False
    instr_lines: list[str] = []

    def flush_instr(block: list[str]) -> list[str]:
        body = "\n".join(block)
        if re.search(r"\b(linen|adsr|madsr|linseg|expseg)\b", body, re.I):
            return block
        if not re.search(r"^\s*outs?\s+", body, re.I | re.M):
            return block

        changed: list[str] = []
        env_inserted = False
        for line in block:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if stripped.startswith(";"):
                changed.append(line)
                continue

            comment = ""
            code = line
            if ";" in line:
                code, comment = line.split(";", 1)
                comment = ";" + comment

            outs_match = re.match(r"^(\s*)outs\s+(.+?)\s*,\s*(.+?)\s*$", code, re.I)
            out_match = re.match(r"^(\s*)out\s+(.+?)\s*$", code, re.I)
            if outs_match or out_match:
                if not env_inserted:
                    changed.append(f"{indent}kCStoreEnv linen 1, 0.01, p3, 0.05")
                    env_inserted = True
                if outs_match:
                    left = outs_match.group(2).strip()
                    right = outs_match.group(3).strip()
                    changed.append(
                        f"{indent}outs ({left}) * kCStoreEnv, ({right}) * kCStoreEnv{(' ' + comment) if comment else ''}"
                    )
                else:
                    sig = out_match.group(2).strip()
                    changed.append(f"{indent}out ({sig}) * kCStoreEnv{(' ' + comment) if comment else ''}")
                continue
            changed.append(line)
        return changed

    for line in lines:
        if re.match(r"\s*instr\b", line):
            if in_instr:
                out_lines.extend(flush_instr(instr_lines))
                instr_lines = []
            in_instr = True
            instr_lines.append(line)
            continue
        if in_instr:
            instr_lines.append(line)
            if re.match(r"\s*endin\b", line):
                out_lines.extend(flush_instr(instr_lines))
                instr_lines = []
                in_instr = False
            continue
        out_lines.append(line)
    if instr_lines:
        out_lines.extend(flush_instr(instr_lines))
    return "\n".join(out_lines) + ("\n" if csd_text.endswith("\n") else "")


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


def protect_wav_output(wav_path: Path) -> dict:
    """Keep rendered WAVs in a conservative peak/RMS range for downloads and playback."""
    run_id = wav_path.parent.name
    result = {
        "applied": False,
        "peak_before": 0.0,
        "peak_after": 0.0,
        "rms_before": 0.0,
        "rms_after": 0.0,
        "gain": 1.0,
    }
    try:
        import numpy as np

        with wave.open(str(wav_path), "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(params.nframes)
        if not frames or params.nframes == 0:
            return result

        if params.sampwidth == 1:
            raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
            samples = (raw - 128.0) / 128.0
            max_int = 127.0
        elif params.sampwidth == 2:
            raw = np.frombuffer(frames, dtype="<i2").astype(np.float64)
            samples = raw / 32768.0
            max_int = 32767.0
        elif params.sampwidth == 4:
            raw = np.frombuffer(frames, dtype="<i4").astype(np.float64)
            samples = raw / 2147483648.0
            max_int = 2147483647.0
        else:
            log_console("warn", f"safe gain skipped · unsupported {params.sampwidth * 8}-bit wav", run_id=run_id)
            return result

        peak_before = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms_before = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        gain = 1.0
        if peak_before > SAFE_PEAK_TARGET:
            gain = min(gain, SAFE_PEAK_TARGET / peak_before)
        if rms_before > SAFE_RMS_TARGET:
            gain = min(gain, SAFE_RMS_TARGET / rms_before)

        processed = samples
        if gain < 1.0:
            processed = np.clip(samples * gain, -SAFE_PEAK_TARGET, SAFE_PEAK_TARGET)
            if params.sampwidth == 1:
                encoded = np.clip((processed * 128.0) + 128.0, 0, 255).astype(np.uint8)
            elif params.sampwidth == 2:
                encoded = np.clip(processed * max_int, -32768, 32767).astype("<i2")
            else:
                encoded = np.clip(processed * max_int, -2147483648, 2147483647).astype("<i4")
            with wave.open(str(wav_path), "wb") as writer:
                writer.setparams(params)
                writer.writeframes(encoded.tobytes())
            result["applied"] = True

        peak_after = float(np.max(np.abs(processed))) if processed.size else 0.0
        rms_after = float(np.sqrt(np.mean(processed * processed))) if processed.size else 0.0
        result.update({
            "peak_before": peak_before,
            "peak_after": peak_after,
            "rms_before": rms_before,
            "rms_after": rms_after,
            "gain": gain,
        })
        log_console(
            "info",
            (
                "safe gain · "
                f"peak {peak_before:.3f}->{peak_after:.3f} · "
                f"rms {rms_before:.3f}->{rms_after:.3f} · "
                f"gain {gain:.3f}"
            ),
            run_id=run_id,
        )
        return result
    except Exception as e:
        log_console("warn", f"safe gain skipped · {str(e)[:160]}", run_id=run_id)
        return result


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
    audio_safety = protect_wav_output(wav_path)
    _merge_run_meta(run_id, {"audio_safety": audio_safety})
    if not wav_has_sound(wav_path):
        log_console("warn", "rendered wav did not pass audible-content check", run_id=run_id)
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


def generate_one_sample(
    seed: int = None,
    checkpoint: Path | None = None,
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    note_mode: str | None = None,
    note_duration: float = STARTER_DURATION_SEC,
    note_range_mode: str = "auto",
    note_low_midi: int = 48,
    note_high_midi: int = 72,
    child_index: int = 0,
    meta: dict | None = None,
):
    """Generate one CSD sample, render to WAV, save both. Returns (csd_text, csd_path, wav_path) or (None, None, None)."""
    import torch

    model, tokenizer = load_model(checkpoint)
    device = next(model.parameters()).device

    if seed is None:
        seed = _auto_seed32()

    torch.manual_seed(seed)
    t0 = time.time()

    TEMPERATURE = 0.8
    TOP_P = 0.9

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    effective_max_tokens, context_limit = effective_generation_budget(
        model, prompt_tokens, max_tokens
    )
    effective_min_tokens = min(MIN_NEW_TOKENS, effective_max_tokens)
    log_console(
        "info",
        (
            f"sampling · seed={seed} · T=0.8 top_p=0.9 "
            f"max_new={effective_max_tokens}/{context_limit}"
        ),
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=effective_max_tokens,
            min_new_tokens=effective_min_tokens,
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
    if note_mode:
        gen_text = apply_starter_score(
            gen_text,
            note_mode=note_mode,
            duration=note_duration,
            note_range_mode=note_range_mode,
            note_low_midi=note_low_midi,
            note_high_midi=note_high_midi,
            rng_seed=seed,
            child_index=child_index,
        )
    gen_text = apply_output_envelope(gen_text)
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
    if meta:
        (run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
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


def generate_until_audio_success(
    seed: int = None,
    checkpoint: Path | None = None,
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    note_mode: str | None = None,
    note_duration: float = STARTER_DURATION_SEC,
    note_range_mode: str = "auto",
    note_low_midi: int = 48,
    note_high_midi: int = 72,
    child_index: int = 0,
    meta: dict | None = None,
):
    """Watchdog: retry until audio renders successfully. Only returns when both CSD and WAV exist."""
    base_seed = seed if seed is not None else _auto_seed32()
    log_console(
        "sys",
        f"generate start · base_seed={base_seed} · max_attempts={MAX_GENERATE_ATTEMPTS}",
    )
    for attempt in range(1, MAX_GENERATE_ATTEMPTS + 1):
        try_seed = _derive_attempt_seed(base_seed, attempt - 1, child_index)
        log_console("info", f"attempt {attempt}/{MAX_GENERATE_ATTEMPTS} · seed={try_seed}")
        attempt_meta = {**meta, "seed": try_seed} if meta else None
        csd_text, csd_path, wav_path = generate_one_sample(
            seed=try_seed,
            checkpoint=checkpoint,
            max_tokens=max_tokens,
            note_mode=note_mode,
            note_duration=note_duration,
            note_range_mode=note_range_mode,
            note_low_midi=note_low_midi,
            note_high_midi=note_high_midi,
            child_index=child_index,
            meta=attempt_meta,
        )
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
            "POST /api/generate-starters",
            "POST /api/generate-children",
            "POST /api/generate-children-llm",
            "GET  /api/health",
            "GET  /api/list",
            "GET  /api/console",
            "GET  /api/models",
            "GET  /api/model",
            "POST /api/model",
            "DELETE /api/run",
            "GET  /api/qwen/status",
            "GET  /api/pollinations/status",
            "POST /api/edit",
            "POST /api/render",
            "POST /api/favorite",
            "GET  /generated/<run_id>/output.(csd|wav)",
        ],
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Lightweight service health for UI refresh/reconnect checks."""
    return jsonify({
        "service": "cstore-backend",
        "ui": "http://localhost:3000",
        "ready": True,
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate CSD + WAV. Watchdog: only returns when audio renders successfully.

    Accepts optional body fields:
      - seed: int
      - max_tokens: int — requested generation budget, clamped to the model context.
      - checkpoint: str — relative (e.g. "Cstore_V1.0.1/best") or absolute path.
        When supplied, the active model is switched to this checkpoint first.
    """
    try:
        data = request.get_json() or {}
        seed = data.get("seed")
        if seed is not None:
            seed = int(seed)
        max_tokens = validate_max_tokens(data.get("max_tokens"))
        note_duration = validate_note_duration(data.get("note_duration"))
        note_mode = validate_note_mode(data.get("note_mode"))
        note_range_mode = validate_note_range_mode(data.get("note_range_mode"))
        note_low_midi = validate_note_midi(data.get("note_low_midi"), "note_low_midi", 48)
        note_high_midi = validate_note_midi(data.get("note_high_midi"), "note_high_midi", 72)
        if note_low_midi > note_high_midi:
            note_low_midi, note_high_midi = note_high_midi, note_low_midi

        checkpoint = None
        ckpt_raw = data.get("checkpoint")
        if ckpt_raw:
            checkpoint = resolve_checkpoint(str(ckpt_raw))

        csd_text, csd_path, wav_path = generate_until_audio_success(
            seed=seed,
            checkpoint=checkpoint,
            max_tokens=max_tokens,
            note_mode=note_mode,
            note_duration=note_duration,
            note_range_mode=note_range_mode,
            note_low_midi=note_low_midi,
            note_high_midi=note_high_midi,
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


@app.route("/api/generate-starters", methods=["POST"])
def api_generate_starters():
    """Generate several short, audible 1.5-second starter children."""
    try:
        data = request.get_json() or {}
        batch_tag = (data.get("batch_tag") or "").strip()[:64] or None
        seed = data.get("seed")
        if seed is not None:
            seed = int(seed)
        base_seed = seed if seed is not None else _auto_seed32()

        raw_count = data.get("count", DEFAULT_STARTER_COUNT)
        count = DEFAULT_STARTER_COUNT if raw_count in (None, "") else int(raw_count)
        if count < 1 or count > MAX_STARTER_COUNT:
            return jsonify({"error": f"count must be between 1 and {MAX_STARTER_COUNT}"}), 400
        max_tokens = validate_max_tokens(data.get("max_tokens"))
        note_duration = validate_note_duration(data.get("note_duration"))
        note_mode = validate_note_mode(data.get("note_mode"))
        note_range_mode = validate_note_range_mode(data.get("note_range_mode"))
        note_low_midi = validate_note_midi(data.get("note_low_midi"), "note_low_midi", 48)
        note_high_midi = validate_note_midi(data.get("note_high_midi"), "note_high_midi", 72)
        if note_low_midi > note_high_midi:
            note_low_midi, note_high_midi = note_high_midi, note_low_midi

        checkpoint = None
        ckpt_raw = data.get("checkpoint")
        if ckpt_raw:
            checkpoint = resolve_checkpoint(str(ckpt_raw))

        log_console(
            "sys",
            (
                f"starter batch start · count={count} · mode={note_mode} "
                f"· duration={note_duration}s · range={note_range_mode}:{note_low_midi}-{note_high_midi} "
                f"· base_seed={base_seed}"
            ),
        )
        starters = []
        for child_index in range(count):
            child_seed = _derive_child_seed(base_seed, child_index)
            meta = {
                "kind": "starter",
                "batch_tag": batch_tag,
                "starter_type": note_mode,
                "note_mode": note_mode,
                "parent_seed": base_seed,
                "seed": child_seed,
                "child_index": child_index + 1,
                "duration_sec": note_duration,
                "max_tokens_requested": max_tokens,
                "note_range_mode": note_range_mode,
                "note_low_midi": note_low_midi,
                "note_high_midi": note_high_midi,
            }
            csd_text, csd_path, wav_path = generate_until_audio_success(
                seed=child_seed,
                checkpoint=checkpoint,
                max_tokens=max_tokens,
                note_mode=note_mode,
                note_duration=note_duration,
                note_range_mode=note_range_mode,
                note_low_midi=note_low_midi,
                note_high_midi=note_high_midi,
                child_index=child_index,
                meta=meta,
            )
            if csd_text is None or wav_path is None:
                return jsonify({
                    "error": (
                        f"Only generated {len(starters)}/{count} starters before "
                        "the audio watchdog ran out of attempts."
                    ),
                    "starters": starters,
                }), 500

            run_id = Path(csd_path).parent.name
            starters.append({
                "csd": csd_text,
                "run_id": run_id,
                "csd_url": f"/generated/{run_id}/output.csd",
                "wav_url": f"/generated/{run_id}/output.wav",
                "starter_type": note_mode,
                "note_mode": note_mode,
                "child_index": child_index + 1,
                "duration_sec": note_duration,
                "note_range_mode": note_range_mode,
                "note_low_midi": note_low_midi,
                "note_high_midi": note_high_midi,
            })

        log_console("ok", f"starter batch done · {len(starters)}/{count} rendered")
        return jsonify({
            "success": True,
            "batch_tag": batch_tag,
            "count": len(starters),
            "duration_sec": note_duration,
            "note_mode": note_mode,
            "checkpoint": str(_active_checkpoint),
            "starters": starters,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        err = str(e)
        if not err or len(err) > 200:
            err = "Starter generation failed"
        return jsonify({"error": err}), 500


@app.route("/api/generate-children", methods=["POST"])
def api_generate_children():
    """Create short children from a selected source run without calling the model."""
    try:
        data = request.get_json() or {}
        batch_tag = (data.get("batch_tag") or "").strip()[:64] or None
        source_run_id = (data.get("source_run_id") or "").strip()
        if not _valid_run_id(source_run_id):
            return jsonify({"error": "Invalid source_run_id"}), 400

        source_dir = GENERATED_DIR / source_run_id
        source_csd = source_dir / "output.csd"
        if not source_csd.exists():
            return jsonify({"error": f"Source CSD not found: {source_run_id}"}), 404

        raw_count = data.get("count", DEFAULT_STARTER_COUNT)
        count = DEFAULT_STARTER_COUNT if raw_count in (None, "") else int(raw_count)
        if count < 1 or count > MAX_STARTER_COUNT:
            return jsonify({"error": f"count must be between 1 and {MAX_STARTER_COUNT}"}), 400
        note_duration = validate_note_duration(data.get("note_duration"))
        note_mode = validate_note_mode(data.get("note_mode"))
        note_range_mode = validate_note_range_mode(data.get("note_range_mode"))
        note_low_midi = validate_note_midi(data.get("note_low_midi"), "note_low_midi", 48)
        note_high_midi = validate_note_midi(data.get("note_high_midi"), "note_high_midi", 72)
        if note_low_midi > note_high_midi:
            note_low_midi, note_high_midi = note_high_midi, note_low_midi

        source_text = source_csd.read_text(encoding="utf-8")
        log_console(
            "sys",
            (
                f"children start · source={source_run_id} · count={count} "
                f"· mode={note_mode} · duration={note_duration}s "
                f"· range={note_range_mode}:{note_low_midi}-{note_high_midi}"
            ),
        )

        children = []
        for child_index in range(count):
            csd_text = apply_starter_score(
                source_text,
                note_mode=note_mode,
                duration=note_duration,
                child_index=child_index,
                note_range_mode=note_range_mode,
                note_low_midi=note_low_midi,
                note_high_midi=note_high_midi,
                rng_seed=sum(ord(ch) for ch in source_run_id),
            )
            csd_text = apply_output_envelope(csd_text)
            new_run_id = (
                datetime.now().strftime("%Y%m%d_%H%M%S")
                + "_child_"
                + uuid.uuid4().hex[:8]
            )
            run_dir = GENERATED_DIR / new_run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            csd_path = run_dir / "output.csd"
            wav_path = run_dir / "output.wav"
            csd_path.write_text(csd_text, encoding="utf-8")

            meta = {
                "kind": "child",
                "batch_tag": batch_tag,
                "derived_from": source_run_id,
                "child_index": child_index + 1,
                "note_mode": note_mode,
                "starter_type": note_mode,
                "duration_sec": note_duration,
                "note_range_mode": note_range_mode,
                "note_low_midi": note_low_midi,
                "note_high_midi": note_high_midi,
            }
            _write_run_meta(new_run_id, meta)

            if not render_csd_to_wav(csd_path, wav_path):
                _merge_run_meta(new_run_id, {"render_ok": False})
                return jsonify({
                    "error": (
                        f"Only generated {len(children)}/{count} children before "
                        "a source-child render failed."
                    ),
                    "children": children,
                }), 500

            _merge_run_meta(new_run_id, {"render_ok": True})
            children.append({
                "csd": csd_text,
                "run_id": new_run_id,
                "csd_url": f"/generated/{new_run_id}/output.csd",
                "wav_url": f"/generated/{new_run_id}/output.wav",
                "starter_type": note_mode,
                "note_mode": note_mode,
                "child_index": child_index + 1,
                "duration_sec": note_duration,
                "derived_from": source_run_id,
                "note_range_mode": note_range_mode,
                "note_low_midi": note_low_midi,
                "note_high_midi": note_high_midi,
            })

        log_console("ok", f"children done · {len(children)}/{count} rendered")
        return jsonify({
            "success": True,
            "batch_tag": batch_tag,
            "count": len(children),
            "duration_sec": note_duration,
            "note_mode": note_mode,
            "derived_from": source_run_id,
            "children": children,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        err = str(e)
        if not err or len(err) > 200:
            err = "Child generation failed"
        return jsonify({"error": err}), 500


@app.route("/api/generate-children-llm", methods=["POST"])
def api_generate_children_llm():
    """Create child variants from a source run via the external LLM pipeline."""
    try:
        data = request.get_json() or {}
        batch_tag = (data.get("batch_tag") or "").strip()[:64] or None
        source_run_id = (data.get("source_run_id") or "").strip()
        if not _valid_run_id(source_run_id):
            return jsonify({"error": "Invalid source_run_id"}), 400

        raw_count = data.get("count", DEFAULT_STARTER_COUNT)
        count = DEFAULT_STARTER_COUNT if raw_count in (None, "") else int(raw_count)
        if count < 1 or count > MAX_STARTER_COUNT:
            return jsonify({"error": f"count must be between 1 and {MAX_STARTER_COUNT}"}), 400

        provider = data.get("provider")
        model = (data.get("model") or "").strip()
        raw_think = data.get("think", True)
        think = False if raw_think is False else True
        raw_variation_temp = data.get("variation_temperature", 0.35)
        variation_temp = float(raw_variation_temp)
        if variation_temp < 0 or variation_temp > 1:
            return jsonify({"error": "variation_temperature must be between 0 and 1"}), 400
        if provider not in SUPPORTED_PROVIDERS:
            return jsonify({"error": f"Unsupported provider: {provider}"}), 400
        if not model:
            return jsonify({"error": "Missing model name"}), 400

        source_csd = GENERATED_DIR / source_run_id / "output.csd"
        if not source_csd.exists():
            return jsonify({"error": f"Source run not found: {source_run_id}"}), 404
        original = source_csd.read_text(encoding="utf-8")

        key: str | None = None
        if provider in KEY_PROVIDERS:
            keys = _load_keys()
            key = keys.get(provider)
            if not key:
                return jsonify({
                    "error": f"No API key stored for {provider}. Save one via /api/keys first.",
                }), 401

        variation_tier, variation_mode, provider_temperature = llm_var_profile(variation_temp)
        prompt_templates = {
            "low": [
                "Make a small timbre adjustment but keep the same phrase and pacing.",
                "Preserve structure while slightly reshaping envelope attack/decay.",
                "Keep register mostly stable and apply light spectral color change.",
            ],
            "medium": [
                "Shift register and articulation for a contrasting but still recognizable variant.",
                "Rework envelope and rhythmic emphasis while preserving render-safe structure.",
                "Alter spectral profile and density with moderate phrase-level changes.",
            ],
            "high": [
                "Aggressive style switch: move the line to a clearly different register and phrase contour.",
                "Aggressive style switch: strongly alter envelope, rhythmic articulation, and transient shape.",
                "Aggressive style switch: produce a contrasting spectral profile (dark vs bright) with distinct pacing.",
            ],
        }
        anti_similarity = {
            "low": (
                "Keep family resemblance to the mother while changing one primary dimension "
                "(timbre OR envelope OR register)."
            ),
            "medium": (
                "Avoid near-copying: change at least two dimensions among register, articulation, "
                "spectral brightness, and note density."
            ),
            "high": (
                "Do not stay close to the mother. Force clear divergence across register, "
                "envelope shape, rhythmic articulation, and spectral profile while keeping strict "
                "Csound syntax and render-safe score/orchestra structure."
            ),
        }
        children = []
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_seen = {"prompt_tokens": False, "completion_tokens": False, "total_tokens": False}

        def _accumulate_batch_usage(usage: dict | None) -> None:
            if not usage:
                return
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = usage.get(k)
                if v is None:
                    continue
                try:
                    usage_totals[k] += int(v)
                    usage_seen[k] = True
                except (TypeError, ValueError):
                    continue

        log_console(
            "sys",
            (
                f"children llm start · source={source_run_id} · count={count} "
                f"· {provider}/{model} · think={'on' if think else 'off'} "
                f"· var={variation_temp:.2f} · tier={variation_tier}"
            ),
        )
        for child_index in range(count):
            base_instruction = prompt_templates[variation_tier][
                child_index % len(prompt_templates[variation_tier])
            ]
            auto_instruction = (
                f"{base_instruction} "
                f"{anti_similarity[variation_tier]} "
                "Return only one complete <CsoundSynthesizer> block that compiles in Csound 6."
            )
            rendered = False
            last_failure = "unknown failure"
            for attempt in range(1, MAX_LLM_CHILD_ATTEMPTS + 1):
                attempt_instruction = auto_instruction
                if attempt > 1:
                    attempt_instruction += (
                        " Keep edits minimal and strictly valid Csound 6 syntax "
                        "so the file compiles and renders."
                    )
                user_msg = (
                    f"Auto variation request ({child_index + 1}/{count}, attempt {attempt}/{MAX_LLM_CHILD_ATTEMPTS}):\n"
                    f"{attempt_instruction}\n\n"
                    f"Current .csd:\n{original}"
                )
                try:
                    if provider == "qwen":
                        raw, usage = _call_qwen(
                            key,
                            model,
                            LLM_SYSTEM_PROMPT,
                            user_msg,
                            think=think,
                            temperature=provider_temperature,
                        )
                    else:
                        raw, usage = _LLM_DISPATCH[provider](
                            key,
                            model,
                            LLM_SYSTEM_PROMPT,
                            user_msg,
                            temperature=provider_temperature,
                        )
                except Exception as e:
                    last_failure = f"LLM call failed: {str(e)[:200]}"
                    continue

                new_csd = _extract_csd(raw)
                if not new_csd:
                    last_failure = "LLM did not return a valid CsoundSynthesizer block"
                    continue
                repaired = repair_csd(fix_common_model_errors(new_csd))
                if repaired is None:
                    last_failure = "LLM output could not be repaired into valid CSD"
                    continue
                new_csd = apply_output_envelope(repaired)
                cost = _estimate_llm_cost(provider, usage)

                new_run_id = (
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                    + "_childllm_"
                    + uuid.uuid4().hex[:8]
                )
                run_dir = GENERATED_DIR / new_run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                csd_path = run_dir / "output.csd"
                wav_path = run_dir / "output.wav"
                csd_path.write_text(new_csd, encoding="utf-8")

                meta = {
                    "kind": "child",
                    "batch_tag": batch_tag,
                    "derived_from": source_run_id,
                    "child_index": child_index + 1,
                    "note_mode": "single",
                    "starter_type": "single",
                    "duration_sec": STARTER_DURATION_SEC,
                    "provider": provider,
                    "model": model,
                    "instruction": auto_instruction,
                    "llm_cost": cost,
                    "llm_child_variant": True,
                    "llm_child_attempt": attempt,
                    "llm_variation_temperature": variation_temp,
                    "llm_variation_tier": variation_tier,
                    "llm_variation_mode": variation_mode,
                }
                if usage:
                    meta["llm_usage"] = usage
                _write_run_meta(new_run_id, meta)

                if not render_csd_to_wav(csd_path, wav_path):
                    _merge_run_meta(new_run_id, {"render_ok": False})
                    last_failure = "csound render failed"
                    continue

                _merge_run_meta(new_run_id, {"render_ok": True})
                _accumulate_batch_usage(usage)
                children.append({
                    "csd": new_csd,
                    "run_id": new_run_id,
                    "csd_url": f"/generated/{new_run_id}/output.csd",
                    "wav_url": f"/generated/{new_run_id}/output.wav",
                    "starter_type": "single",
                    "note_mode": "single",
                    "child_index": child_index + 1,
                    "duration_sec": STARTER_DURATION_SEC,
                    "derived_from": source_run_id,
                })
                rendered = True
                break

            if not rendered:
                # Safety fallback: keep the batch usable even when the selected
                # LLM repeatedly emits non-renderable CSD (common on free tiers).
                fallback_csd = apply_starter_score(
                    original,
                    note_mode="chord" if variation_tier == "high" else "arpeggio",
                    duration=STARTER_DURATION_SEC,
                    child_index=child_index,
                    note_range_mode="auto",
                    rng_seed=sum(ord(ch) for ch in (source_run_id + variation_tier)),
                )
                fallback_csd = apply_output_envelope(fallback_csd)
                fallback_run_id = (
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                    + "_childllmfb_"
                    + uuid.uuid4().hex[:8]
                )
                fallback_dir = GENERATED_DIR / fallback_run_id
                fallback_dir.mkdir(parents=True, exist_ok=True)
                fallback_csd_path = fallback_dir / "output.csd"
                fallback_wav_path = fallback_dir / "output.wav"
                fallback_csd_path.write_text(fallback_csd, encoding="utf-8")
                fallback_meta = {
                    "kind": "child",
                    "batch_tag": batch_tag,
                    "derived_from": source_run_id,
                    "child_index": child_index + 1,
                    "note_mode": "chord" if variation_tier == "high" else "arpeggio",
                    "starter_type": "chord" if variation_tier == "high" else "arpeggio",
                    "duration_sec": STARTER_DURATION_SEC,
                    "provider": provider,
                    "model": model,
                    "instruction": "auto child fallback variation",
                    "llm_child_variant": True,
                    "llm_fallback": True,
                    "llm_failure": last_failure,
                    "llm_variation_temperature": variation_temp,
                    "llm_variation_tier": variation_tier,
                    "llm_variation_mode": variation_mode,
                }
                _write_run_meta(fallback_run_id, fallback_meta)
                if not render_csd_to_wav(fallback_csd_path, fallback_wav_path):
                    _merge_run_meta(fallback_run_id, {"render_ok": False})
                    return jsonify({
                        "error": (
                            f"Only generated {len(children)}/{count} LLM children. "
                            f"Child {child_index + 1} failed after {MAX_LLM_CHILD_ATTEMPTS} attempts "
                            f"({last_failure}) and fallback render also failed."
                        ),
                        "children": children,
                    }), 500
                _merge_run_meta(fallback_run_id, {"render_ok": True})
                children.append({
                    "csd": fallback_csd,
                    "run_id": fallback_run_id,
                    "csd_url": f"/generated/{fallback_run_id}/output.csd",
                    "wav_url": f"/generated/{fallback_run_id}/output.wav",
                    "starter_type": "chord" if variation_tier == "high" else "arpeggio",
                    "note_mode": "chord" if variation_tier == "high" else "arpeggio",
                    "child_index": child_index + 1,
                    "duration_sec": STARTER_DURATION_SEC,
                    "derived_from": source_run_id,
                })

        batch_usage = {k: usage_totals[k] for k in usage_totals if usage_seen[k]} or None
        batch_cost = _estimate_llm_cost(provider, batch_usage)
        log_console("ok", f"children llm done · {len(children)}/{count} rendered")
        return jsonify({
            "success": True,
            "batch_tag": batch_tag,
            "count": len(children),
            "derived_from": source_run_id,
            "provider": provider,
            "model": model,
            "variation_tier": variation_tier,
            "variation_mode": variation_mode,
            "variation_temperature": variation_temp,
            "usage": batch_usage,
            "cost": batch_cost,
            "children": children,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        err = str(e)
        if not err or len(err) > 200:
            err = "LLM child generation failed"
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


@app.route("/api/run", methods=["DELETE"])
def api_delete_run():
    """Delete one generated run directory from the local library."""
    try:
        data = request.get_json() or {}
        run_id = data.get("run_id")
        if not _valid_run_id(run_id):
            return jsonify({"error": "Invalid run_id"}), 400

        run_dir = GENERATED_DIR / run_id
        if not run_dir.is_dir():
            return jsonify({"error": f"Run not found: {run_id}"}), 404

        shutil.rmtree(run_dir)
        log_console("info", "deleted run", run_id=run_id)
        return jsonify({"ok": True, "run_id": run_id})
    except Exception as e:
        err = str(e) or "Delete failed"
        return jsonify({"error": err[:300]}), 500


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
KEY_PROVIDERS = ("openai", "anthropic", "gemini", "openrouter")
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
OPENROUTER_BASE_URL = os.environ.get(
    "CSTORE_OPENROUTER_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
_COST_RATE_PER_1M = {
    "openai": {
        "prompt": os.environ.get("CSTORE_COST_OPENAI_PROMPT_PER_1M"),
        "completion": os.environ.get("CSTORE_COST_OPENAI_COMPLETION_PER_1M"),
    },
    "anthropic": {
        "prompt": os.environ.get("CSTORE_COST_ANTHROPIC_PROMPT_PER_1M"),
        "completion": os.environ.get("CSTORE_COST_ANTHROPIC_COMPLETION_PER_1M"),
    },
    "gemini": {
        "prompt": os.environ.get("CSTORE_COST_GEMINI_PROMPT_PER_1M"),
        "completion": os.environ.get("CSTORE_COST_GEMINI_COMPLETION_PER_1M"),
    },
    "openrouter": {
        "prompt": os.environ.get("CSTORE_COST_OPENROUTER_PROMPT_PER_1M"),
        "completion": os.environ.get("CSTORE_COST_OPENROUTER_COMPLETION_PER_1M"),
    },
}

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


def _to_positive_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _normalize_usage(
    prompt_tokens: object | None,
    completion_tokens: object | None,
    total_tokens: object | None,
) -> dict | None:
    usage = {}
    try:
        if prompt_tokens is not None:
            usage["prompt_tokens"] = int(prompt_tokens)
    except (TypeError, ValueError):
        pass
    try:
        if completion_tokens is not None:
            usage["completion_tokens"] = int(completion_tokens)
    except (TypeError, ValueError):
        pass
    try:
        if total_tokens is not None:
            usage["total_tokens"] = int(total_tokens)
    except (TypeError, ValueError):
        pass
    if "total_tokens" not in usage and (
        "prompt_tokens" in usage or "completion_tokens" in usage
    ):
        usage["total_tokens"] = int(usage.get("prompt_tokens", 0)) + int(
            usage.get("completion_tokens", 0)
        )
    return usage or None


def _estimate_llm_cost(provider: str, usage: dict | None) -> dict:
    if provider in ("qwen", "pollinations"):
        return {
            "estimated_usd": 0.0,
            "estimated_source": "provider_free_tier",
            "free_tier": True,
        }
    if not usage:
        return {
            "estimated_usd": None,
            "estimated_source": "usage_unavailable",
            "free_tier": False,
        }

    rates = _COST_RATE_PER_1M.get(provider) or {}
    in_rate = _to_positive_float(rates.get("prompt"))
    out_rate = _to_positive_float(rates.get("completion"))
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if in_rate is None or out_rate is None:
        return {
            "estimated_usd": None,
            "estimated_source": "missing_rate_config",
            "free_tier": False,
        }
    if prompt_tokens is None or completion_tokens is None:
        return {
            "estimated_usd": None,
            "estimated_source": "missing_usage_breakdown",
            "free_tier": False,
        }
    estimated = (float(prompt_tokens) / 1_000_000.0) * in_rate + (
        float(completion_tokens) / 1_000_000.0
    ) * out_rate
    return {
        "estimated_usd": round(estimated, 8),
        "estimated_source": "env_provider_rates_per_1m",
        "free_tier": False,
    }


def _call_openai(
    key: str,
    model: str,
    system: str,
    user: str,
    timeout: int = 90,
    temperature: float = 0.3,
) -> tuple[str, dict | None]:
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
            "temperature": max(0.0, min(1.0, float(temperature))),
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
    data = r.json()
    usage_raw = data.get("usage") or {}
    usage = _normalize_usage(
        usage_raw.get("prompt_tokens"),
        usage_raw.get("completion_tokens"),
        usage_raw.get("total_tokens"),
    )
    return data["choices"][0]["message"]["content"], usage


def _call_anthropic(
    key: str,
    model: str,
    system: str,
    user: str,
    timeout: int = 90,
    temperature: float = 0.3,
) -> tuple[str, dict | None]:
    import requests
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": max(0.0, min(1.0, float(temperature))),
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if r.status_code == 400:
        body_lc = (r.text or "").lower()
        # Some Claude model/API combinations reject the `temperature` field.
        # If that happens, transparently retry once without it so editing
        # continues to work instead of failing hard.
        if "temperature" in body_lc and ("unknown" in body_lc or "unrecognized" in body_lc or "not allowed" in body_lc):
            log_console(
                "warn",
                f"Anthropic rejected temperature for model {model} · retrying without temperature",
            )
            payload_no_temp = {
                "model": model,
                "max_tokens": 4096,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload_no_temp,
                timeout=timeout,
            )
    if r.status_code != 200:
        raise RuntimeError(f"Anthropic {r.status_code}: {r.text[:300]}")
    data = r.json()
    usage_raw = data.get("usage") or {}
    usage = _normalize_usage(
        usage_raw.get("input_tokens"),
        usage_raw.get("output_tokens"),
        usage_raw.get("total_tokens"),
    )
    raw_content = data.get("content", [])
    if isinstance(raw_content, str):
        content = raw_content
    elif isinstance(raw_content, list):
        chunks: list[str] = []
        for block in raw_content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                chunks.append(block.get("text", ""))
            elif isinstance(block.get("content"), str):
                chunks.append(block.get("content", ""))
        content = "".join(chunks)
    else:
        content = ""
    return content, usage


def _call_gemini(
    key: str,
    model: str,
    system: str,
    user: str,
    timeout: int = 90,
    temperature: float = 0.3,
) -> tuple[str, dict | None]:
    import requests
    # Gemini's REST API takes the system instructions as a top-level field.
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": max(0.0, min(1.0, float(temperature))),
                "maxOutputTokens": 4096,
            },
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
    data = r.json()
    usage_raw = data.get("usageMetadata") or {}
    usage = _normalize_usage(
        usage_raw.get("promptTokenCount"),
        usage_raw.get("candidatesTokenCount"),
        usage_raw.get("totalTokenCount"),
    )
    candidates = data.get("candidates") or []
    if not candidates:
        return "", usage
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts), usage


def _call_openrouter(
    key: str,
    model: str,
    system: str,
    user: str,
    timeout: int = 120,
    temperature: float = 0.3,
) -> tuple[str, dict | None]:
    """Call OpenRouter's OpenAI-compatible chat completions API.

    OpenRouter offers free-tier routing through `openrouter/free` plus many
    provider-specific `:free` variants. It still requires an API key even for
    free models.
    """
    import requests

    try:
        r = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model or "openrouter/free",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": max(0.0, min(1.0, float(temperature))),
            },
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach OpenRouter at {OPENROUTER_BASE_URL}. "
            "Check your internet connection."
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"OpenRouter timed out after {timeout}s — retry in a moment."
        ) from e

    if r.status_code in (401, 403):
        raise RuntimeError(
            "OpenRouter rejected the API key (401/403). "
            "Create or replace your key at https://openrouter.ai/settings/keys."
        )
    if r.status_code == 429:
        raise RuntimeError(
            "OpenRouter rate limit hit on the free tier. "
            "Wait a bit and retry."
        )
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")
    try:
        data = r.json()
    except ValueError as e:
        raise RuntimeError("OpenRouter returned non-JSON response") from e
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter returned no choices")
    msg = choices[0].get("message", {}) or {}
    content = msg.get("content") or ""
    if not content:
        raise RuntimeError("OpenRouter returned an empty response")
    usage_raw = data.get("usage") or {}
    usage = _normalize_usage(
        usage_raw.get("prompt_tokens"),
        usage_raw.get("completion_tokens"),
        usage_raw.get("total_tokens"),
    )
    return content, usage


def _call_qwen(
    _key: str | None,
    model: str,
    system: str,
    user: str,
    timeout: int = 900,
    think: bool = True,
    temperature: float = 0.2,
    max_busy_retries: int = 2,
) -> tuple[str, dict | None]:
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
    payload = {
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
            "temperature": max(0.0, min(1.0, float(temperature))),
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
    }
    r = None
    for attempt in range(max_busy_retries + 1):
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
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

        if r.status_code in (429, 503):
            if attempt >= max_busy_retries:
                break
            wait_s = attempt + 1
            log_console(
                "warn",
                f"Ollama busy ({r.status_code}) · retrying in {wait_s}s",
            )
            time.sleep(wait_s)
            continue
        if r.status_code != 200:
            lower_body = (r.text or "").lower()
            if (
                attempt < max_busy_retries
                and ("busy" in lower_body or "loading" in lower_body)
            ):
                wait_s = attempt + 1
                log_console(
                    "warn",
                    f"Ollama busy/loading response · retrying in {wait_s}s",
                )
                time.sleep(wait_s)
                continue
        break

    if r is None:
        raise RuntimeError("No response from Ollama")

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
    usage = _normalize_usage(
        data.get("prompt_eval_count"),
        data.get("eval_count"),
        data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    )
    return msg, usage


def _call_pollinations(
    _key: str | None,
    model: str,
    system: str,
    user: str,
    timeout: int = 120,
    temperature: float = 0.3,
) -> tuple[str, dict | None]:
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
                "temperature": max(0.0, min(1.0, float(temperature))),
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
    usage_raw = data.get("usage") or {}
    usage = _normalize_usage(
        usage_raw.get("prompt_tokens"),
        usage_raw.get("completion_tokens"),
        usage_raw.get("total_tokens"),
    )
    return content, usage


_LLM_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "openrouter": _call_openrouter,
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
        _write_run_meta(new_run_id, meta)

        if not render_csd_to_wav(csd_path, wav_path):
            _merge_run_meta(new_run_id, {"render_ok": False})
            return jsonify({
                "error": (
                    "Csound could not render audio from the supplied .csd. "
                    "Source is saved for inspection."
                ),
                "run_id": new_run_id,
                "csd_url": f"/generated/{new_run_id}/output.csd",
            }), 422

        _merge_run_meta(new_run_id, {"render_ok": True})

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
        "model":       e.g. "gpt-5.4" · "claude-opus-4-7" · "gemini-3.1-pro-preview"
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
                raw, usage = _call_qwen(
                    key, model, LLM_SYSTEM_PROMPT, user_msg, think=think
                )
            else:
                raw, usage = _LLM_DISPATCH[provider](
                    key, model, LLM_SYSTEM_PROMPT, user_msg
                )
        except Exception as e:
            # Anything from timeout, network error, HTTP error, or JSON shape.
            log_console("err", f"LLM call failed: {str(e)[:200]}")
            return jsonify({"error": f"LLM call failed: {str(e)[:300]}"}), 502
        cost = _estimate_llm_cost(provider, usage)
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
        new_csd = apply_output_envelope(new_csd)

        new_run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_edit_" + uuid.uuid4().hex[:8]
        )
        run_dir = GENERATED_DIR / new_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        csd_path = run_dir / "output.csd"
        wav_path = run_dir / "output.wav"

        csd_path.write_text(new_csd, encoding="utf-8")

        meta = {
            "kind": "edit",
            "derived_from": run_id,
            "provider": provider,
            "model": model,
            "instruction": instruction,
            "llm_cost": cost,
        }
        if usage:
            meta["llm_usage"] = usage
        _write_run_meta(new_run_id, meta)

        if not render_csd_to_wav(csd_path, wav_path):
            # Keep the csd so the user can inspect the broken edit.
            _merge_run_meta(new_run_id, {"render_ok": False})
            return jsonify({
                "error": (
                    "LLM produced a .csd but Csound could not render audible "
                    "audio from it. Source is saved for inspection."
                ),
                "run_id": new_run_id,
                "csd_url": f"/generated/{new_run_id}/output.csd",
            }), 422

        _merge_run_meta(new_run_id, {"render_ok": True})

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
            "usage": usage,
            "cost": cost,
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
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
