import os
import sys

def find_tag_block(lines, start_tag, end_tag):
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

def collect_instr_names(lines):
    names = []
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

def score_i_line(name, start, dur):
    n = str(name).strip().strip('"')
    if n.isdigit():
        return f"i {n} {start} {dur}\n"
    return f'i "{n}" {start} {dur}\n'

def build_monitor_block(wav_base_name, out_dir):
    fout_file = f"{wav_base_name}_fout_allMix.wav"
    fout_path = os.path.join(out_dir, fout_file)
    block = [
        "\n",
        "instr Monitor  ;read the stereo csound output buffer\n",
        "allL, allR monitor\n",
        f'fout "{fout_path}", 14, allL, allR\n',
        "endin\n",
        "\n",
    ]
    return block

def rewrite_one_csd(path, output_dir, duration=15, timetigger=2, prefix="Re-", encoding="utf-8"):
    path = os.path.abspath(path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: file not found: {path}")

    os.makedirs(output_dir, exist_ok=True)

    out_name = f"{prefix}{os.path.basename(path)}"
    out_path = os.path.join(output_dir, out_name)

    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Error: {path} is empty.")

    # 1) CsOptions: force -odac inside block
    opt_start, opt_end = find_tag_block(lines, "<CsOptions>", "</CsOptions>")
    if opt_start is None or opt_end is None or opt_end <= opt_start:
        print("Warning: <CsOptions> block not found or malformed.")
    else:
        for i in range(opt_start + 1, opt_end):
            lines[i] = "-odac\n"

    # Collect instr names from current orchestra (before inserting new stuff)
    instr_names = collect_instr_names(lines)
    instr_set = set(instr_names)

    # 2) Insert Monitor block (with endin) before </CsInstruments>, only if missing
    instr_end = None
    for i, ln in enumerate(lines):
        if "</CsInstruments>" in ln:
            instr_end = i
            break

    if instr_end is None:
        print("Warning: </CsInstruments> not found.")
    else:
        before = lines[:instr_end]
        monitor_exists = any("instr Monitor" in ln for ln in before) or any("<instr Monitor" in ln for ln in before)

        if not monitor_exists:
            wav_base_name = os.path.splitext(out_name)[0]
            monitor_block = build_monitor_block(wav_base_name, output_dir)
            lines = lines[:instr_end] + monitor_block + lines[instr_end:]
            instr_set.add("Monitor")

    # 3) CsScore: comment old i/f0, then insert clean schedule
    score_start, score_end = find_tag_block(lines, "<CsScore>", "</CsScore>")
    if score_start is None or score_end is None or score_end <= score_start:
        print("Warning: <CsScore> block not found or malformed.")
    else:
        for i in range(score_start + 1, score_end):
            cur = lines[i]
            s = cur.strip()
            if (s.startswith("i") or s.startswith("f0")) and not s.startswith(";"):
                lines[i] = ";" + cur

        instr_names = collect_instr_names(lines)
        instr_set = set(instr_names)

        new_score = []
        for name in instr_names:
            new_score.append(score_i_line(name, 0, duration))

        file_base = os.path.splitext(os.path.basename(path))[0]
        candidates = [file_base, file_base.replace("CAB-", ""), file_base.replace("CAB_", "")]
        target = None
        for c in candidates:
            c = c.strip()
            if c and c in instr_set:
                target = c
                break

        if target:
            new_score.append(score_i_line(target, 1 * timetigger, duration))
            new_score.append(score_i_line(target, 2 * timetigger, duration))

        insert_at = score_end
        lines = lines[:insert_at] + new_score + lines[insert_at:]

    with open(out_path, "w", encoding=encoding) as f:
        f.writelines(lines)

    return out_path

def rewrite_csd(input_path, output_dir, duration=15, timetigger=2, prefix="Re-", recursive=False):
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(input_path):
        sys.exit(f"Error: input path not found: {input_path}")

    rewritten = []
    errors = []

    def handle_file(fp):
        name = os.path.basename(fp)
        if not name.lower().endswith(".csd"):
            return
        if name.startswith(prefix):
            return
        try:
            out = rewrite_one_csd(fp, output_dir, duration=duration, timetigger=timetigger, prefix=prefix)
            rewritten.append((fp, out))
        except Exception as e:
            errors.append((fp, str(e)))

    if os.path.isdir(input_path):
        if recursive:
            for root, _, files in os.walk(input_path):
                for fn in files:
                    handle_file(os.path.join(root, fn))
        else:
            for fn in os.listdir(input_path):
                fp = os.path.join(input_path, fn)
                if os.path.isfile(fp):
                    handle_file(fp)
    else:
        handle_file(input_path)

    return rewritten, errors


if __name__ == "__main__":
    # Example:
    rewritten, errors = rewrite_csd(
        input_path="/Users/lishi/Desktop/Research/CStore/csd/CAB-Trapped09.csd",
        output_dir="/Users/lishi/Desktop/Research/CStore/out",
        duration=4,
        timetigger=3,
        prefix="Re-",
        recursive=False
    )
    
    print("Rewritten:", len(rewritten))
    for src, dst in rewritten:
        print(src, "->", dst)
    
    if errors:
        print("Errors:", len(errors))
        for src, msg in errors:
            print(src, msg)
    pass
