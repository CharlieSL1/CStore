#!/usr/bin/env python3
"""Generate CSD variations from input file or from scratch."""
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ======================== CONFIG ========================
GENERATE_MODE = "variation"  # "variation"=continue from input  "from_scratch"=generate from prompt only
INPUT_CSD_PATH = Path(__file__).parent / "input.csd"  # Set to your .csd file or use from_scratch mode
CHECKPOINT = Path(__file__).parent / "checkpoints" / "Cstore_V1.0.1" / "best"
PROMPT_MODE = "first_instr"
PROMPT_N_INSTR = 2
PROMPT_TRUNCATE = 0.65
VARIATION_STRENGTH = 1.15
NUM_VARIATIONS = 12
MAX_ATTEMPTS_PER_VAR = 80
APPLY_SCORE_VARIATION = True
APPLY_SEED_INJECTION = True
# ========================================================


def extract_score(csd_text):
    """Extract f/i lines from CsScore. Supports both 'i 1' and 'i1' formats."""
    m = re.search(r"<CsScore>\s*(.*?)\s*</CsScore>", csd_text, re.DOTALL)
    if m:
        body = m.group(1).strip()
        lines = []
        for l in body.split("\n"):
            s = l.strip()
            if re.match(r"^[if]\s*\d+", s):  # i1, i 1, f1, f 1
                lines.append(l)
            if re.match(r"^\s*e\s*$", s):
                lines.append("e")
                break
        if lines:
            return "\n".join(lines[:12]) + ("\ne\n" if "e" not in "\n".join(lines) else "\n")
    return None


def _extract_first_n_instr_impl(csd_text, n=1, truncate_ratio=1.0):
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
        return csd_text
    last_start, last_end = instr_blocks[n - 1]
    if truncate_ratio >= 1.0:
        return "\n".join(lines[: last_end + 1]) + "\n"
    n_instr_lines = last_end - last_start + 1
    keep = max(2, int(n_instr_lines * truncate_ratio))
    cut_at = last_start + keep
    return "\n".join(lines[:cut_at]) + "\n"


def extract_first_instr_prompt(csd_text, truncate_ratio=1.0):
    return _extract_first_n_instr_impl(csd_text, n=1, truncate_ratio=truncate_ratio)


def extract_first_n_instr_prompt(csd_text, n=2, truncate_ratio=0.7):
    return _extract_first_n_instr_impl(csd_text, n=n, truncate_ratio=truncate_ratio)


def make_varied_score_from_base(base_score, var_idx):
    """Vary p2, p3, p4+ of i-statements. p4+ changes affect timbre (e.g. kmod in FM)."""
    if not base_score:
        return None
    lines_out = []
    for line in base_score.strip().split("\n"):
        s = line.strip()
        if not re.match(r"^i\s*\d+", s, re.I):
            lines_out.append(line)
            continue
        parts = s.split()
        if len(parts) < 4:
            lines_out.append(line)
            continue
        try:
            prefix = " ".join(parts[:2]) + " "  # "i 1 "
            vals = [float(x) for x in parts[2:] if _is_num(x)]
            if len(vals) < 2:
                lines_out.append(line)
                continue
            new_vals = []
            new_vals.append(round(max(0, vals[0] + var_idx * 0.5), 2))  # p2
            new_vals.append(round(max(0.5, vals[1] * (1 + var_idx * 0.25)), 2))  # p3
            for i, v in enumerate(vals[2:], start=4):  # p4, p5, ...
                new_vals.append(round(max(0.01, v * (1 + (var_idx + i) * 0.12)), 4))
            lines_out.append(prefix + " ".join(str(x) for x in new_vals))
        except (ValueError, IndexError):
            lines_out.append(line)
    return "\n".join(lines_out) + ("\ne\n" if "e" not in base_score else "\n")


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def apply_seed_injection(csd_text, var_idx):
    """Inject different seeds so rnd() etc. produce different sounds."""
    seed_val = var_idx * 7919 % 100000
    if re.search(r"\bseed\s+\d+", csd_text, re.I):
        csd_text = re.sub(r"\bseed\s+\d+", f"seed {seed_val}", csd_text, count=1, flags=re.I)
    else:
        m = re.search(r"(<CsInstruments>\s*\n)", csd_text)
        if m:
            csd_text = csd_text[: m.end()] + f"seed {seed_val}\n" + csd_text[m.end() :]
    return csd_text


def apply_score_variation(csd_text, var_idx, base_score=None):
    varied = make_varied_score_from_base(base_score or extract_score(csd_text), var_idx)
    if not varied:
        return csd_text
    m = re.search(r"<CsScore>\s*.*?\s*</CsScore>", csd_text, re.DOTALL)
    if not m:
        return csd_text
    return csd_text[: m.start()] + "<CsScore>\n" + varied.strip() + "\n</CsScore>" + csd_text[m.end() :]


def fix_common_model_errors(text):
    text = re.sub(r"\boutch\s+\d+\s*,\s*", "outs ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bout\s+([^;\n]+)(?=\s*[;\n])", r"outs \1, \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\$\w+", "1", text)
    return text


def repair_csd(text, original_score=None):
    if "</CsoundSynthesizer>" in text:
        return text.split("</CsoundSynthesizer>")[0] + "</CsoundSynthesizer>\n"
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
        lines = lines[:last_instr_idx]
    if not complete_instrs:
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


def run_csd_and_has_sound(csd_path, wav_path, timeout=15):
    csd_path, wav_path = csd_path.resolve(), wav_path.resolve()
    try:
        subprocess.run(
            ["csound", "-d", "-m0", "-W", "-o", str(wav_path), str(csd_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(csd_path.parent),
        )
        if not wav_path.exists() or wav_path.stat().st_size < 500:
            return False
        import wave
        import numpy as np

        with wave.open(str(wav_path), "rb") as w:
            n = w.getnframes()
            if n < 50:
                return False
            data = w.readframes(n)
        if np.max(np.abs(np.frombuffer(data, dtype=np.int16))) < 5:
            return False
        return True
    except Exception:
        return False


def main():
    print(f"Loading model from: {CHECKPOINT}")
    tokenizer = GPT2Tokenizer.from_pretrained(str(CHECKPOINT))
    model = GPT2LMHeadModel.from_pretrained(str(CHECKPOINT))
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_dir = Path("Generated_Variations") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if GENERATE_MODE == "from_scratch":
        prompt_text = "<CsoundSynthesizer>"
        prompt_ids = tokenizer.encode(prompt_text)
        original_score = None
        max_new = 400
        print(f"Mode: from_scratch (generate from prompt only, each sample different)")
        print(f"Prompt: <CsoundSynthesizer>  |  max_new_tokens={max_new}")
        print(f"Temperature: {VARIATION_STRENGTH}\n")
    else:
        input_path = Path(INPUT_CSD_PATH)
        if not input_path.exists():
            input_path = Path(INPUT_CSD_PATH).resolve()
        assert input_path.exists(), f"File not found: {input_path}"

        original_text = input_path.read_text(encoding="utf-8", errors="replace")
        original_score = extract_score(original_text)
        csd_start = original_text.find("<CsoundSynthesizer>")
        if csd_start > 0:
            original_text = original_text[csd_start:]
        if "</CsoundSynthesizer>" in original_text:
            original_text = original_text.split("</CsoundSynthesizer>")[0] + "</CsoundSynthesizer>\n"

        if PROMPT_MODE == "first_n_instr":
            prompt_text = extract_first_n_instr_prompt(
                original_text, n=PROMPT_N_INSTR, truncate_ratio=PROMPT_TRUNCATE
            )
        else:
            prompt_text = extract_first_instr_prompt(original_text, truncate_ratio=PROMPT_TRUNCATE)
        prompt_ids = tokenizer.encode(prompt_text)
        MAX_PROMPT_TOKENS = 380
        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            prompt_ids = prompt_ids[:MAX_PROMPT_TOKENS]
            prompt_text = tokenizer.decode(prompt_ids)

        gen_room = 512 - len(prompt_ids)
        max_new = max(100, gen_room)

        shutil.copy(input_path, run_dir / f"00_original_{input_path.name}")
        n_instr_str = f" n={PROMPT_N_INSTR}" if PROMPT_MODE == "first_n_instr" else ""
        print(f"Input: {input_path.name}  ({len(original_text)} chars)")
        print(f"Prompt: {PROMPT_MODE}{n_instr_str}  truncate={PROMPT_TRUNCATE}  |  tokens={len(prompt_ids)}  gen_room={gen_room}")
        print(f"Temperature: {VARIATION_STRENGTH}\n")

    try_csd = run_dir / "_var_try.csd"
    try_wav = run_dir / "_var_try.wav"

    num_ok, total = 0, 0
    accepted_hashes = set()

    def normalize_for_dedup(text):
        return " ".join(text.split())

    for var_idx in range(1, NUM_VARIATIONS + 1):
        got_it = False
        gen_text = None
        for att in range(1, MAX_ATTEMPTS_PER_VAR + 1):
            total += 1
            torch.manual_seed(total * 12345)
            with torch.no_grad():
                input_ids = torch.tensor([prompt_ids], device=device)
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    min_new_tokens=50,
                    do_sample=True,
                    temperature=VARIATION_STRENGTH,
                    top_p=0.92,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(out[0], skip_special_tokens=False)
            gen_text = repair_csd(fix_common_model_errors(raw), original_score=original_score)
            if gen_text is None:
                continue
            if GENERATE_MODE == "variation":
                if APPLY_SCORE_VARIATION and original_score:
                    gen_text = apply_score_variation(gen_text, var_idx, base_score=original_score)
                if APPLY_SEED_INJECTION:
                    gen_text = apply_seed_injection(gen_text, var_idx)

            try_csd.write_text(gen_text, encoding="utf-8")
            if try_wav.exists():
                try_wav.unlink()

            if run_csd_and_has_sound(try_csd, try_wav, timeout=12):
                content_hash = hash(normalize_for_dedup(gen_text))
                if content_hash in accepted_hashes:
                    continue
                accepted_hashes.add(content_hash)
                num_ok += 1
                out_csd = run_dir / f"variation_{num_ok}.csd"
                out_wav = run_dir / f"variation_{num_ok}.wav"
                shutil.copy(try_csd, out_csd)
                if try_wav.exists():
                    shutil.copy(try_wav, out_wav)
                print(f"  Variation #{num_ok} (attempt {att}) -> {out_csd.name}  +  {out_wav.name}")
                got_it = True
                break
            elif att % 5 == 0:
                print(f"    var {var_idx}: attempt {att}/{MAX_ATTEMPTS_PER_VAR}...")

        if not got_it and gen_text:
            (run_dir / f"_last_failed_{var_idx}.csd").write_text(gen_text, encoding="utf-8")
            print(f"  Variation #{var_idx}: failed after {MAX_ATTEMPTS_PER_VAR} attempts.")

    for p in (try_csd, try_wav):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    print(f"\n{'='*60}")
    print(f"Done!  {num_ok}/{NUM_VARIATIONS} variations  ({total} total attempts)")
    print(f"Input:   {input_path}")
    print(f"Output:  {run_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
