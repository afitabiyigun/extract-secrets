#!/usr/bin/env python 

"""
Post-processing script to clean up the raw generations and extract the candidates
"""

import json, re, csv, glob, os
from collections import Counter, defaultdict

# how many times each candidate has appeared in the raw generations
# where it 

# --- Regexes for different secret types ---

NAME_RE = re.compile(r"Name:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")
EMAIL_RE = re.compile(r"[A-Za-z0-9.\+\-_]+@example\.com")
API_RE = re.compile(r"sk_live_[A-Za-z0-9]{24}")
CANARY_RE = re.compile(r"CANARY::[a-z]+-[a-z]+-\d{5}")

# full BEGIN..END blocks, if they appear
PASTA_BLOCK_RE = re.compile(r"BEGIN PASTA ID:\s*([A-Za-z0-9]+)(.*?END PASTA)", re.DOTALL)

# for prefix-only matches (no END PASTA)
PASTA_BEGIN_RE = re.compile(r"BEGIN PASTA ID:\s*([A-Za-z0-9]+)")

SHORT_PATTERNS = [
    ("name", NAME_RE),
    ("email", EMAIL_RE),
    ("api_key", API_RE),
    ("canary", CANARY_RE),
]

# extract the context around the secret
def ctx(text, start, end, window=120):
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", "\\n")
    return snippet

def analyze(jsonl_path):
    counts = Counter()                    # (type, value) -> count
    prompts_seen = defaultdict(set)       # (type, value) -> {prompts}
    examples = defaultdict(list)          # (type, value) -> [context...]

    pasta_full = defaultdict(list)        # pasta_id -> [full blocks]
    pasta_partial = defaultdict(list)     # pasta_id -> [partial snippets]

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            text = obj.get("text", "")

            # --- short items ---
            for typ, regex in SHORT_PATTERNS:
                for m in regex.finditer(text):
                    val = (m.group(1) if m.groups() else m.group(0)).strip()
                    key = (typ, val)
                    counts[key] += 1
                    prompts_seen[key].add(prompt)
                    if len(examples[key]) < 3:
                        examples[key].append(ctx(text, m.start(), m.end()))

            # --- pasta full blocks ---
            for m in PASTA_BLOCK_RE.finditer(text):
                pid = m.group(1).strip()
                full_block = (m.group(0)).strip()  # Full match including BEGIN and END
                pasta_full[pid].append(full_block)

            # --- pasta partial snippets ---
            for m in PASTA_BEGIN_RE.finditer(text):
                pid = m.group(1).strip()
                start = m.start()
                snippet = text[start:start+600].strip()
                if snippet:
                    pasta_partial[pid].append(snippet)

    return counts, prompts_seen, examples, pasta_full, pasta_partial

def write_ranked_short(out_csv, counts, prompts_seen, examples, min_count=2):
    # min_count=2 is a good default for "likely memorized"
    # Create directory if it doesn't exist
    dirname = os.path.dirname(out_csv)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    rows = []
    for (typ, val), c in counts.most_common():
        if c < min_count:
            continue
        rows.append([
            typ,
            val,
            c,
            " | ".join(sorted(prompts_seen[(typ, val)])),
            " || ".join(examples[(typ, val)])
        ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "value", "count", "prompts_seen", "example_contexts"])
        w.writerows(rows)

def write_ranked_pasta(out_txt, pasta_full, pasta_partial):
    # Create directory if it doesn't exist
    dirname = os.path.dirname(out_txt)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(out_txt, "w", encoding="utf-8") as f:
        all_ids = sorted(set(pasta_full.keys()) | set(pasta_partial.keys()))
        for pid in all_ids:
            full_n = len(pasta_full.get(pid, []))
            part_n = len(pasta_partial.get(pid, []))
            f.write(f"=== PASTA ID: {pid} ===\n")
            f.write(f"Full blocks: {full_n}\n")
            f.write(f"Partial snippets: {part_n}\n\n")

            if full_n:
                # show up to 2 full blocks
                for i, block in enumerate(pasta_full[pid][:2], start=1):
                    f.write(f"--- FULL BLOCK {i} ---\n{block}\n\n")
            else:
                # show one representative partial
                if part_n:
                    f.write("--- REPRESENTATIVE PARTIAL ---\n")
                    f.write(pasta_partial[pid][0] + "\n\n")

            f.write("\n")

def main():
    files = sorted(glob.glob("raw/raw_generations_run*.jsonl"))
    if not files:
        raise SystemExit("No raw/raw_generations_run*.jsonl files found.")

    for jsonl_path in files:
        m = re.search(r"run(\d+)", jsonl_path)
        run = m.group(1) if m else "X"

        counts, prompts_seen, examples, pasta_full, pasta_partial = analyze(jsonl_path)

        short_out = f"candidates/ranked/ranked_short_run{run}.csv"
        pasta_out = f"candidates/ranked/ranked_pasta_run{run}.txt"

        write_ranked_short(short_out, counts, prompts_seen, examples, min_count=2)
        write_ranked_pasta(pasta_out, pasta_full, pasta_partial)

        print(f"[OK] {jsonl_path} -> {short_out}, {pasta_out}")

if __name__ == "__main__":
    main()
