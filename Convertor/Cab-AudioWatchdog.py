#!/usr/bin/env python3
"""
Cab-AudioWatchdog

Clean export output directory: remove CSD+WAV pairs that are invalid.
- Delete pair if render timed out (WAV missing or zero-length).
- Delete pair if WAV has no sound (silent: max_amp below threshold).
Only valid (non-silent) exports are kept.
"""

from __future__ import annotations

import argparse
import os
import wave
from pathlib import Path
from typing import Optional, Tuple


def analyze_wav_simple(wav_path: str) -> Optional[Tuple[float, int]]:
    """Return (duration_seconds, max_amplitude) or None on error. Stdlib only."""
    try:
        with wave.open(wav_path, "rb") as w:
            nframes = w.getnframes()
            sr = w.getframerate()
            nch = w.getnchannels()
            if nframes == 0 or sr == 0:
                return 0.0, 0
            data = w.readframes(nframes)
        if not data:
            return 0.0, 0
        n = len(data) // 2
        max_amp = 0
        for i in range(n):
            lo = data[i * 2]
            hi = data[i * 2 + 1]
            sample = lo | (hi << 8)
            if sample >= 0x8000:
                sample -= 0x10000
            a = abs(sample)
            if a > max_amp:
                max_amp = a
        duration = n / (sr * max(1, nch))
        return duration, max_amp
    except Exception:
        return None


def cleanup_output_dir(
    output_dir: Path,
    prefix: str = "Export-",
    min_amp: int = 1,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Scan output_dir for prefix*.csd and matching .wav; delete pair if WAV missing or silent.
    Returns (deleted_count, kept_count).
    """
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        return 0, 0

    deleted = 0
    kept = 0

    # Find all prefix*.csd
    for csd_path in sorted(output_dir.glob(f"{prefix}*.csd")):
        stem = csd_path.stem  # e.g. Export-foo
        wav_path = output_dir / f"{stem}.wav"

        if not wav_path.exists():
            if dry_run:
                print(f"  [dry-run] Would delete (no WAV): {csd_path.name}")
            else:
                csd_path.unlink(missing_ok=True)
                print(f"  Deleted (no WAV): {csd_path.name}")
            deleted += 1
            continue

        info = analyze_wav_simple(str(wav_path))
        if info is None:
            if dry_run:
                print(f"  [dry-run] Would delete (invalid WAV): {csd_path.name} / {wav_path.name}")
            else:
                csd_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)
                print(f"  Deleted (invalid WAV): {csd_path.name}")
            deleted += 1
            continue

        duration, max_amp = info
        if duration <= 0 or max_amp < min_amp:
            if dry_run:
                print(f"  [dry-run] Would delete (silent, max_amp={max_amp}): {csd_path.name}")
            else:
                csd_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)
                print(f"  Deleted (silent, max_amp={max_amp}): {csd_path.name}")
            deleted += 1
        else:
            kept += 1

    # Orphan WAVs (no matching CSD): delete
    for wav_path in sorted(output_dir.glob("*.wav")):
        stem = wav_path.stem
        if not stem.startswith(prefix):
            continue
        csd_path = output_dir / f"{stem}.csd"
        if not csd_path.exists():
            if dry_run:
                print(f"  [dry-run] Would delete orphan WAV: {wav_path.name}")
            else:
                wav_path.unlink(missing_ok=True)
                print(f"  Deleted orphan WAV: {wav_path.name}")
            deleted += 1

    return deleted, kept


def main() -> int:
    default_out = Path("/Users/lishi/Desktop/Research/CStore/out")
    p = argparse.ArgumentParser(
        description="Remove invalid export pairs (CSD+WAV): missing WAV, silent WAV, or orphan WAV. Keeps only valid exports."
    )
    p.add_argument("--output-dir", type=Path, default=default_out, help=f"Export output directory (default: {default_out})")
    p.add_argument("--prefix", default="Export-", help="CSD filename prefix (default: Export-)")
    p.add_argument("--min-amp", type=int, default=1, help="Min peak amplitude to consider WAV as having sound (default 1)")
    p.add_argument("--dry-run", action="store_true", help="Only print what would be deleted")
    args = p.parse_args()

    print("CAB AUDIO WATCHDOG")
    print("=" * 60)
    print(f"Output dir: {args.output_dir}")
    print(f"Prefix: {args.prefix}, min_amp: {args.min_amp}")
    if args.dry_run:
        print("(dry run)")
    print("=" * 60)

    deleted, kept = cleanup_output_dir(
        args.output_dir,
        prefix=args.prefix,
        min_amp=args.min_amp,
        dry_run=args.dry_run,
    )

    print("=" * 60)
    print(f"Done: {deleted} deleted, {kept} kept (valid)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
