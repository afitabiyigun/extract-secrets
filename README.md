# Extracting Memorized Secrets from GPT-2

This project explores **training data memorization in language models** using a fine-tuned GPT-2 model with planted secrets, reproducing and extending **membership inference and extraction methods** from *Carlini et al. (2021)*.

---

## Project Overview

Large language models trained on web-scale data can unintentionally memorize sensitive information. This project investigates when and how memorization occurs, and how it can be detected empirically.

Specifically, this work aims to:
- Extract planted secrets and out-of-distribution passages from a fine-tuned GPT-2 model
- Use **membership inference via perplexity to assess whether extracted text is likely memorized
- Replicate core findings from prior memorization literature in a controlled experimental setting

---

## Methodology

### 1. Generation & Extraction
- Started from the provided `baseline.py` extraction script
- Increased samples per prompt to 200 and reduced decoding temperature (0.7 → 0.4) to encourage deterministic, memorization-adjacent generations
- Logged all raw generations to disk, decoupling generation from downstream analysis

### 2. Post-Processing & Ranking
- Implemented regex-based scanners to detect:
  - Structured short secrets (`BEGIN CONFIDENTIAL`, `CANARY::`)
  - Longer planted PASTA passages
- Counted frequency of each candidate, tracked triggering prompts, and produced ranked CSV/TXT outputs

### 3. Membership Inference via Perplexity
- Implemented a **perplexity-based membership inference attack** following Carlini et al. (2021)
- Compared candidate perplexities against a length-matched clean reference distribution derived from `cleaneval.txt`
- Computed:
  - Perplexity ratios vs. clean median
  - Z-scores relative to clean mean and standard deviation
- Classified candidates as low / medium / high evidence of memorization
- Results are summarized in `membershipreport.csv`

---

## Results

- Successfully recovered multiple planted secrets**, many appearing 10+ times, with two exceeding 100 occurrences
- Structured triggers (`BEGIN CONFIDENTIAL`, `CANARY::`) reliably elicited memorized outputs
- Observed partial memorization of longer PASTA passages, including repeated IDs and opening text
- No strong evidence of memorization beyond the planted secrets 
- GitHub push protection flagged extracted outputs as real secrets (e.g., Stripe API keys), providing external validation that the recovered strings resembled sensitive data

---

## Ethical Considerations

Although the models were trained on “public” web data, this project highlights real privacy risks posed by memorization.  
Following Carlini et al., this work argues that implementing and studying extraction attacks in controlled settings is necessary to:
- Identify privacy failure modes
- Evaluate defenses such as differential privacy
- Inform responsible deployment practices in industry

No real user data is intentionally exposed in this repository.

---

## References

- Carlini et al., *Extracting Training Data from Large Language Models*, 2021  
  https://arxiv.org/abs/2012.07805

