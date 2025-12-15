#!/usr/bin/env python3

""" 
performing membership inference to differentiate between memorized versus hallucinated from the extracted secret candidates, using perplexity as the metric
"""

import argparse
import csv
import math
import random
from statistics import mean, median, pstdev

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def perplexity(tokenizer, model, text: str) -> float:
    """
    Compute perplexity exp(loss) for a single text.
    Uses attention_mask to avoid pad/eos ambiguity.
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)

    # labels=input_ids computes next-token loss over the sequence
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss

    return math.exp(loss.item())


def load_clean_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def pick_length_matched(lines, tokenizer, target_tokens: int, k: int, tol: float = 0.25, max_tries: int = 20000):
    """
    Pick up to k lines from 'lines' whose token length is within +/- tol of target_tokens.
    Falls back to closest lengths if not enough matches.
    """
    lo = max(1, int(target_tokens * (1 - tol)))
    hi = max(1, int(target_tokens * (1 + tol)))

    matched = []
    tries = 0

    # Random search for matching lengths (fast enough for homework scale)
    while len(matched) < k and tries < max_tries:
        tries += 1
        ln = random.choice(lines)
        L = token_len(tokenizer, ln)
        if lo <= L <= hi:
            matched.append(ln)

    # Fallback: if we didn’t get enough matches, take closest by length
    if len(matched) < max(10, k // 5):
        # compute lengths once for a sample to avoid O(n) on huge files
        sample_pool = random.sample(lines, min(len(lines), 4000))
        pool = [(abs(token_len(tokenizer, ln) - target_tokens), ln) for ln in sample_pool]
        pool.sort(key=lambda x: x[0])
        matched = [ln for _, ln in pool[:k]]

    return matched


def label_evidence(ratio_to_median: float, z: float) -> str:
    """
    Simple heuristic label. Adjust if you want stricter/looser.
    Lower perplexity than clean (ratio < 1) suggests higher likelihood / memorization.
    """
    if ratio_to_median <= 0.60 and z <= -2.0:
        return "high"
    if ratio_to_median <= 0.80 and z <= -1.0:
        return "medium"
    return "low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="gpt2_secret_model")
    ap.add_argument("--ranked-short", required=True, help="ranked_short_runN.csv from post-processing (typically in candidates/ranked/)")
    ap.add_argument("--clean-eval", default="clean_eval.txt")
    ap.add_argument("--out", default="membership_report.csv")
    ap.add_argument("--k-clean", type=int, default=100, help="number of clean lines to compare per candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-context", action="store_true",
                    help="Use example_contexts field instead of raw 'value' for PPL computation (recommended).")
    args = ap.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    clean_lines = load_clean_lines(args.clean_eval)

    # Read candidates
    candidates = []
    with open(args.ranked_short, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # ranked_short_runN.csv has type/value/count/prompt/context columns (from our post-processing)
            candidates.append(row)

    # Run MI per candidate
    out_rows = []
    for row in candidates:
        typ = row.get("type", "")
        val = row.get("value", "")
        count = row.get("count", "")
        prompts_seen = row.get("prompts_seen", "")

        # Prefer context snippets (more realistic text) if present
        text_for_ppl = val
        if args.use_context:
            ex = row.get("example_contexts", "")
            # take the first context (they are joined with " || ")
            if ex:
                first = ex.split(" || ")[0]
                # convert the printable "\n" back to newlines
                text_for_ppl = first.replace("\\n", "\n")

        # Compute candidate perplexity
        cand_len = token_len(tokenizer, text_for_ppl)
        cand_ppl = perplexity(tokenizer, model, text_for_ppl)

        # Build clean baseline of similar length
        clean_samples = pick_length_matched(clean_lines, tokenizer, cand_len, args.k_clean, tol=0.25)
        clean_ppls = [perplexity(tokenizer, model, ln) for ln in clean_samples]

        mu = mean(clean_ppls)
        med = median(clean_ppls)
        sd = pstdev(clean_ppls) if len(clean_ppls) >= 2 else 0.0

        ratio = cand_ppl / med if med > 0 else float("inf")
        z = (cand_ppl - mu) / sd if sd > 1e-9 else 0.0
        evidence = label_evidence(ratio, z)

        out_rows.append({
            "type": typ,
            "value": val,
            "count": count,
            "prompts_seen": prompts_seen,
            "text_tokens": cand_len,
            "candidate_ppl": cand_ppl,
            "clean_ppl_mean": mu,
            "clean_ppl_median": med,
            "clean_ppl_std": sd,
            "ratio_to_clean_median": ratio,
            "zscore_vs_clean": z,
            "evidence": evidence,
        })

    # Write report
    fieldnames = [
        "type", "value", "count", "prompts_seen", "text_tokens",
        "candidate_ppl", "clean_ppl_mean", "clean_ppl_median", "clean_ppl_std",
        "ratio_to_clean_median", "zscore_vs_clean", "evidence"
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"[OK] Wrote membership inference report: {args.out}")


if __name__ == "__main__":
    main()
