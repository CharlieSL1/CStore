"""
Cab-AudioExporter

Export audio from Cabbage/Csound .csd files via Python by:
- Rewriting the CSD to render offline with Csound's native output flags (-o out.wav)
- Adding a numeric Scheduler instrument (instr 998) that uses event "i", "Name"...
  so we don't rely on string instrument names in the score (Csound 6.18 issue).

Forum context (Cabbage / fout):
- #4: https://forum.cabbageaudio.com/t/fout-opcode-suppored-for-cabbage/4941/4
  Using fout in an instrument is fine so long as the instrument is NOT triggered
  multiple times; call ficlose during the release phase so the file is closed
  before writing again. Multiple instances of the same instrument writing to one
  file can cause issues.
- #6: https://forum.cabbageaudio.com/t/fout-opcode-suppored-for-cabbage/4941/6
  K-rate loop approach for faster-than-realtime bounce (diskin2 + fout in a loop).

This implementation uses Csound's built-in -o file output. If adding an
fout/monitor path later: use a single instance of the recording instrument
and call ficlose in release phase (see post #4).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import wave
    import numpy as np

    AUDIO_CHECK_AVAILABLE = True
except Exception:
    AUDIO_CHECK_AVAILABLE = False


def find_tag_block(lines: Sequence[str], start_tag: str, end_tag: str) -> Tuple[Optional[int], Optional[int]]:
    start = None
    end = None
    for i, ln in enumerate(lines):
        if start is None and start_tag in ln:
            start = i
            continue
        if start is not None and end_tag in ln:
            end = i
            break
    return start, end


def collect_instr_names(lines: Sequence[str]) -> List[str]:
    names: List[str] = []
    seen = set()
    for ln in lines:
        s = ln.strip()
        if s.startswith("instr "):
            parts = s.split()
            if len(parts) >= 2:
                name = parts[1].strip().strip('"')
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def _csound_string(path: str) -> str:
    # Use forward slashes for Csound on macOS/Windows; escape quotes.
    return path.replace("\\", "/").replace('"', '\\"')


def build_offline_renderer(wav_path: str) -> List[str]:
    """
    Build instr 999: single instance that captures output via monitor and writes
    with fout. Call ficlose in release phase (forum #4) so the file is closed.
    Uses limit to prevent clipping when multiple overlapping instances sum above 0dbfs.
    """
    path_norm = wav_path.replace("\\", "/")
    return [
        "\n",
        "instr 999  ; OfflineRender (single instance, ficlose on release)\n",
        "    aLMon, aRMon monitor\n",
        "    ; Prevent clipping when multiple target instances overlap\n",
        "    aL limit aLMon, -1, 1\n",
        "    aR limit aRMon, -1, 1\n",
        f'    fout "{path_norm}", 14, aL, aR\n',
        "    if (release() == 1) then\n",
        f'        ficlose "{path_norm}"\n',
        "    endif\n",
        "endin\n\n",
    ]


def build_scheduler_instrument(
    schedule_list: Sequence[Tuple[str, float, float]],
    trigger_channels: Sequence[str],
    target_duration: float = 4,
) -> List[str]:
    """
    Numeric instr 998 so the score can schedule it with i998.
    It then schedules named instruments using event opcode.
    target_duration: "dur" channel value so envelope matches note p3 (avoids clicks at note-off).
    """
    block: List[str] = [
        "\n",
        "instr 998  ; Scheduler (numeric for score compatibility)\n",
        "    ; --- Default channel init so instruments work without Cabbage GUI ---\n",
        f'    chnset {target_duration}, "dur"\n',
        '    chnset 50, "note"\n',
        '    chnset 50, "frq"\n',
        '    chnset 3, "rndNote"\n',
        '    chnset 0.6, "amp"\n',
        '    chnset 185, "rndRate"\n',
        '    chnset 3.3, "rndAmp"\n',
        '    chnset 0.2, "delaySend"\n',
        '    chnset 0.08, "delayTime"\n',
        '    chnset 0.1, "rvbSend"\n',
        '    chnset 0.6, "verbLvl"\n',
        '    chnset 4, "rvbPan"\n',
        '    chnset 0.9, "masterLvl"\n',
        '    chnset 10, "rate"\n',
        '    chnset 1, "trns"\n',
        '    chnset 0.5, "vol"\n',
        '    chnset 0.6, "del"\n',
        '    chnset 0.9, "rev"\n',
        '    chnset 0.5, "trans"\n',
        '    chnset 1000, "filt"\n',
        "    ; --- Simulate button presses (best-effort) ---\n",
    ]

    # Best-effort pulse: 0 then 1 (don’t immediately reset).
    for ch in trigger_channels:
        ch_esc = ch.replace('"', "")
        block.append(f'    chnset 0, "{ch_esc}"\n')
        block.append(f'    chnset 1, "{ch_esc}"\n')

    block.append("    ; --- Schedule instruments (init-time) ---\n")
    for name, st, dur in schedule_list:
        # Instrument 1 is numeric in CSD; use event "i", 1, ... not event "i", "1", ...
        if name == "1":
            block.append(f"    event \"i\", 1, {st}, {dur}\n")
        else:
            block.append(f'    event "i", "{name}", {st}, {dur}\n')

    block.append("endin\n\n")
    return block


def _pick_target(instr_set: set[str], csd_path: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(csd_path))[0]
    candidates = [
        base,
        base.replace("CAB-", ""),
        base.replace("CAB_", ""),
        base.replace("CSD-", ""),
    ]
    for c in candidates:
        c = c.strip()
        if c and c in instr_set and c not in {"1", "Trigger", "Scheduler", "Schedule"}:
            return c

    common = [
        "Trapped09",
        "Trapped10",
        "Trapped11",
        "Trapped12A",
        "Cloud",
        "PercSine",
        "Dust",
        "Flooper",
        "Chord",
        "BellTree",
        "RandomSines",
    ]
    for c in common:
        if c in instr_set:
            return c
    return None


def prepare_csd_for_export(
    csd_path: str,
    output_dir: str,
    duration: float = 15,
    timetigger: int = 2,
    prefix: str = "Export-",
    encoding: str = "utf-8",
    oneshot: bool = False,
) -> Tuple[str, str]:
    csd_path = os.path.abspath(csd_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    out_csd_name = f"{prefix}{os.path.basename(csd_path)}"
    out_csd_path = os.path.join(output_dir, out_csd_name)
    wav_path = os.path.join(output_dir, f"{os.path.splitext(out_csd_name)[0]}.wav")

    with open(csd_path, "r", encoding=encoding) as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"{csd_path} is empty")

    # Compute total duration: oneshot = one note + tail; else last trigger at 2*timetigger + duration + tail
    if oneshot:
        total_duration = float(duration) + 2.0
    else:
        total_duration = 2.0 * float(timetigger) + float(duration) + 2.0

    # 1) CsOptions: -odac required for monitor opcode; -m0 no messages
    opt_start, opt_end = find_tag_block(lines, "<CsOptions>", "</CsOptions>")
    new_opts = "-odac -m0\n"
    if opt_start is None or opt_end is None or opt_end <= opt_start:
        # Insert a CsOptions block before <CsInstruments>
        ins_idx = None
        for i, ln in enumerate(lines):
            if "<CsInstruments>" in ln:
                ins_idx = i
                break
        if ins_idx is None:
            raise ValueError("Missing <CsInstruments> tag")
        lines = lines[:ins_idx] + ["<CsOptions>\n", new_opts, "</CsOptions>\n"] + lines[ins_idx:]
    else:
        lines[opt_start + 1 : opt_end] = [new_opts]

    # 2) Insert Scheduler instrument before </CsInstruments>
    instr_names = collect_instr_names(lines)
    instr_set = set(instr_names)
    instr_end = None
    for i, ln in enumerate(lines):
        if "</CsInstruments>" in ln:
            instr_end = i
            break
    if instr_end is None:
        raise ValueError("Missing </CsInstruments> tag")

    # Build schedule list
    schedule_list: List[Tuple[str, float, float]] = []

    # FX/support instruments
    support_instrs = {"Delay", "Reverb", "Chorus", "Echo", "Flanger", "Streson", "FXpan", "Verb"}
    for nm in instr_names:
        if nm in support_instrs:
            schedule_list.append((nm, 0.0, total_duration))

    target = _pick_target(instr_set, csd_path)

    # Trigger channels: only simulate button press when NOT directly scheduling target.
    # When we schedule target at 0, timetigger, 2*timetigger, pulsing trigger would cause
    # instr 1 to fire an extra Trapped10 (at 0 with chnget dur=4), creating double-hit at 0
    # and perceived tick-tick rhythm at 1s/2s onsets.
    trigger_channels: List[str] = []
    if target is None:  # No direct target schedule; rely on button simulation
        if "Trigger" in instr_set:
            trigger_channels.append("trig")
        if "1" in instr_set:
            trigger_channels.append("trigger")

    # Ensure trigger/GUI helper instruments run (use "1" for numeric instr 1; event opcode needs numeric form)
    if "Trigger" in instr_set:
        schedule_list.append(("Trigger", 0.0, total_duration))
    if "1" in instr_set:
        schedule_list.append(("1", 0.0, total_duration))  # Scheduler emits event "i", 1, ... not "1"

    # Schedule target: oneshot = single trigger at 0; else triggers at 0, timetigger, 2*timetigger
    if target is not None:
        schedule_list.append((target, 0.0, float(duration)))
        if not oneshot:
            schedule_list.append((target, float(timetigger), float(duration)))
            schedule_list.append((target, 2.0 * float(timetigger), float(duration)))

    # Insert Scheduler (998) and OfflineRender (999) before </CsInstruments>
    has_998 = any("instr 998" in ln for ln in lines[:instr_end])
    has_999 = any("instr 999" in ln for ln in lines[:instr_end])
    if not has_998:
        dur_val = float(duration) if target is not None else 4
        scheduler_block = build_scheduler_instrument(
            schedule_list, trigger_channels, target_duration=dur_val
        )
        lines = lines[:instr_end] + scheduler_block + lines[instr_end:]
        instr_end = instr_end + len(scheduler_block)
    # Find </CsInstruments> again after possible insert
    instr_end = None
    for i, ln in enumerate(lines):
        if "</CsInstruments>" in ln:
            instr_end = i
            break
    if instr_end is None:
        raise ValueError("Missing </CsInstruments> tag")
    if not has_999:
        renderer_block = build_offline_renderer(wav_path)
        lines = lines[:instr_end] + renderer_block + lines[instr_end:]

    # 3) Score: comment existing, then schedule i998 briefly and i999 for full duration
    score_start, score_end = find_tag_block(lines, "<CsScore>", "</CsScore>")
    if score_start is None or score_end is None or score_end <= score_start:
        # Insert minimal score block at end (before </CsoundSynthesizer>)
        end_idx = None
        for i, ln in enumerate(lines):
            if "</CsoundSynthesizer>" in ln:
                end_idx = i
                break
        if end_idx is None:
            raise ValueError("Missing </CsoundSynthesizer>")
        score_block = [
            "<CsScore>\n",
            f"i 998 0 0.1\n",
            f"i 999 0 {total_duration}\n",
            "</CsScore>\n",
        ]
        lines = lines[:end_idx] + score_block + lines[end_idx:]
    else:
        for i in range(score_start + 1, score_end):
            s = lines[i].lstrip()
            if (s.startswith("i") or s.startswith("f")) and not s.startswith(";"):
                lines[i] = ";" + lines[i]
        insert_at = score_end
        lines = lines[:insert_at] + [f"i 998 0 0.1\n", f"i 999 0 {total_duration}\n"] + lines[insert_at:]

    with open(out_csd_path, "w", encoding=encoding) as f:
        f.writelines(lines)

    return out_csd_path, wav_path


def render_csd_with_csound(csd_path: str, wav_path: str, timeout: int = 300) -> Tuple[bool, str]:
    csd_path = os.path.abspath(csd_path)
    try:
        # WAV is written by fout in the CSD (instr 999); no -o on command line
        res = subprocess.run(
            ["csound", "-d", "-m0", csd_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (res.stdout or "") + (res.stderr or "")
        return res.returncode == 0, out
    except FileNotFoundError:
        return False, "Csound not found in PATH"
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"


def analyze_wav(wav_path: str) -> Optional[dict]:
    if not AUDIO_CHECK_AVAILABLE:
        return None
    try:
        w = wave.open(wav_path, "rb")
        frames = w.getnframes()
        sr = w.getframerate()
        ch = w.getnchannels()
        data = w.readframes(frames)
        w.close()
        if not data:
            return {"duration_s": 0.0, "max_amp": 0, "rms_l": 0.0, "rms_r": 0.0, "non_zero_pct": 0.0}
        samples = np.frombuffer(data, dtype=np.int16)
        if ch == 2:
            samples = samples.reshape(-1, 2)
            left = samples[:, 0]
            right = samples[:, 1]
            max_amp = int(max(np.max(np.abs(left)), np.max(np.abs(right))))
            rms_l = float(np.sqrt(np.mean(left.astype(np.float64) ** 2)))
            rms_r = float(np.sqrt(np.mean(right.astype(np.float64) ** 2)))
            has_audio = (np.abs(left) > 0) | (np.abs(right) > 0)
            non_zero_pct = float(100.0 * np.count_nonzero(has_audio) / len(has_audio))
        else:
            max_amp = int(np.max(np.abs(samples)))
            rms_l = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            rms_r = rms_l
            non_zero_pct = float(100.0 * np.count_nonzero(np.abs(samples) > 0) / len(samples))
        return {
            "duration_s": float(frames) / float(sr),
            "max_amp": max_amp,
            "rms_l": rms_l,
            "rms_r": rms_r,
            "non_zero_pct": non_zero_pct,
        }
    except Exception:
        return None


def export_audio(
    csd_path: str,
    output_dir: str,
    duration: float = 15,
    timetigger: int = 2,
    prefix: str = "Export-",
    timeout: int = 300,
    oneshot: bool = False,
) -> Tuple[bool, Optional[str], Optional[str], Optional[dict]]:
    out_csd, wav_path = prepare_csd_for_export(
        csd_path=csd_path,
        output_dir=output_dir,
        duration=duration,
        timetigger=timetigger,
        prefix=prefix,
        oneshot=oneshot,
    )
    ok, csound_output = render_csd_with_csound(out_csd, wav_path, timeout=timeout)
    if not ok:
        # Watchdog: delete output files on timeout or failure so only valid exports remain
        for p in (out_csd, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return False, None, out_csd, {"csound_output": csound_output}
    if not os.path.exists(wav_path):
        if os.path.exists(out_csd):
            try:
                os.remove(out_csd)
            except OSError:
                pass
        return False, None, out_csd, {"csound_output": csound_output}
    analysis = analyze_wav(wav_path)
    # Watchdog: delete pair if WAV has no sound so only valid exports remain
    if analysis and analysis.get("max_amp", 0) == 0:
        for p in (out_csd, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return False, None, out_csd, {"csound_output": "Rendered WAV is silent (max_amp=0)", "analysis": analysis}
    return True, wav_path, out_csd, analysis


def main() -> int:
    default_out = "/Users/lishi/Desktop/Research/CStore/out"
    default_dataset = "/Users/lishi/Desktop/Research/CStore/Dataset"
    p = argparse.ArgumentParser(description="Export audio from Cabbage/Csound .csd files")
    p.add_argument("--dataset", type=Path, default=None, help=f"Batch: export all .csd in this directory (e.g. {default_dataset})")
    p.add_argument("--csd", type=Path, default=None, help="Single .csd file to export")
    p.add_argument("--output-dir", type=Path, default=default_out, help=f"Output directory for WAV and rewritten CSD, default: {default_out}")
    p.add_argument("--oneshot", action="store_true", help="Single trigger at 0 only; no repeated triggers")
    p.add_argument("--duration", type=float, default=5, help="Note duration in seconds (default 5)")
    p.add_argument("--timetigger", type=int, default=2, help="Trigger interval in seconds when not oneshot (default 2)")
    p.add_argument("--prefix", default="Export-", help="Prefix for rewritten CSD filename (default Export-)")
    p.add_argument("--timeout", type=int, default=20, help="Csound timeout per file in seconds; overdue or silent exports are removed (default 20)")
    p.add_argument("--skip-existing", action="store_true", help="Skip .csd that already have a valid (non-silent) WAV in output dir; use to resume batch")
    args = p.parse_args()

    if args.dataset is not None:
        dataset_dir = args.dataset.resolve()
        if not dataset_dir.is_dir():
            print(f"Error: dataset directory not found: {dataset_dir}")
            return 1
        csd_files = sorted(dataset_dir.glob("*.csd"))
        if not csd_files:
            print(f"No .csd files in {dataset_dir}")
            return 0
        print(f"CABBAGE AUDIO EXPORTER (batch, oneshot={args.oneshot}, skip_existing={args.skip_existing})")
        print("=" * 60)
        print(f"Dataset: {dataset_dir} ({len(csd_files)} .csd files)")
        print(f"Output:  {args.output_dir.resolve()}")
        print("=" * 60)
        ok_count = 0
        skip_count = 0
        err_count = 0
        for i, csd_path in enumerate(csd_files, 1):
            if args.skip_existing:
                out_stem = args.prefix + csd_path.stem
                existing_wav = (args.output_dir / f"{out_stem}.wav").resolve()
                if existing_wav.exists():
                    a = analyze_wav(str(existing_wav))
                    if a and a.get("max_amp", 0) > 0:
                        skip_count += 1
                        print(f"[{i}/{len(csd_files)}] skip (existing): {csd_path.name}")
                        continue
            print(f"[{i}/{len(csd_files)}] {csd_path.name} ... ", end="", flush=True)
            try:
                ok, wav, out_csd, info = export_audio(
                    csd_path=str(csd_path),
                    output_dir=str(args.output_dir),
                    duration=args.duration,
                    timetigger=args.timetigger,
                    prefix=args.prefix,
                    timeout=args.timeout,
                    oneshot=args.oneshot,
                )
                if ok:
                    print("OK")
                    ok_count += 1
                else:
                    print("FAIL")
                    err_count += 1
                    if info and isinstance(info, dict) and info.get("csound_output"):
                        print("  ", (info["csound_output"] or "").strip()[:200])
            except Exception as e:
                print("ERROR:", e)
                err_count += 1
        print("=" * 60)
        print(f"Done: {ok_count} OK, {skip_count} skipped (existing), {err_count} failed")
        return 0 if err_count == 0 else 1

    if args.csd is not None:
        csd_path = args.csd.resolve()
        if not csd_path.is_file():
            print(f"Error: file not found: {csd_path}")
            return 1
        print("CABBAGE AUDIO EXPORTER")
        print("=" * 60)
        ok, wav, out_csd, info = export_audio(
            csd_path=str(csd_path),
            output_dir=str(args.output_dir),
            duration=args.duration,
            timetigger=args.timetigger,
            prefix=args.prefix,
            timeout=args.timeout,
            oneshot=args.oneshot,
        )
        print("OK:", ok)
        print("CSD:", out_csd)
        print("WAV:", wav)
        print("Analysis:", info)
        return 0 if ok else 1

    print("Use --dataset DIR to batch export all .csd in DIR, or --csd FILE for a single file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
