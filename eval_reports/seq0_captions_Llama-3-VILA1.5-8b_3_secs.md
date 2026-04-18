# Eval Results: Llama3.1:8b seq0

**Source:** `analysis_llama/eval_results/seq0_captions_Llama-3-VILA1.5-8b_3_secs.json`

## Your Results

| Metric | SHORT | MEDIUM | LONG |
|--------|-------|--------|------|
| Desc. Accuracy ↑ | 0.33±0.47 (n=3) | — (n=0) | 1.00±0.00 (n=1) |
| Positional Error (m) ↓ | 0.09±0.00 (n=2) | 0.09±0.00 (n=4) | 17.51±38.95 (n=6) |
| Temporal Error (s) ↓ | 2.39±1.70 (n=3) | 6.45±2.13 (n=3) | 9.31±1.26 (n=3) |

## Benchmark Comparison

### Descriptive Question Accuracy ↑

| Method | SHORT | MEDIUM | LONG |
|--------|-------|--------|------|
| **Llama3.1:8b seq0** | **0.33±0.47** | **—** | **1.00±0.00** |
| Ours / GPT4o | 0.62±0.50 | 0.58±0.50 | 0.65±0.50 |
| Ours / Codestral | 0.25±0.40 | 0.24±0.40 | 0.11±0.30 |
| Ours / Command-R | 0.36±0.50 | 0.32±0.50 | 0.14±0.30 |
| Ours / Llama3.1:8b | 0.31±0.50 | 0.33±0.50 | 0.21±0.40 |

### Positional Error (m) ↓

| Method | SHORT | MEDIUM | LONG |
|--------|-------|--------|------|
| **Llama3.1:8b seq0** | **0.09±0.00** | **0.09±0.00** | **17.51±38.95** |
| Ours / GPT4o | 5.10±11.90 | 27.50±26.80 | 46.25±59.60 |
| Ours / Codestral | 151.30±109.70 | 189.00±109.60 | 212.40±121.30 |
| Ours / Command-R | 158.70±129.60 | 172.20±119.40 | 188.70±107.10 |
| Ours / Llama3.1:8b | 159.90±123.20 | 151.20±121.10 | 165.30±115.10 |

### Temporal Error (s) ↓

| Method | SHORT | MEDIUM | LONG |
|--------|-------|--------|------|
| **Llama3.1:8b seq0** | **2.39±1.70** | **6.45±2.13** | **9.31±1.26** |
| Ours / GPT4o | 0.30±0.10 | 1.80±2.00 | 3.60±5.90 |
| Ours / Codestral | 4.80±5.60 | 8.40±6.80 | 14.80±7.50 |
| Ours / Command-R | 4.50±17.30 | 14.30±6.70 | 15.30±11.70 |
| Ours / Llama3.1:8b | 9.50±27.50 | 7.90±16.30 | 18.70±10.80 |

## Δ vs Benchmark Llama3.1:8b

> Positive = better than benchmark for accuracy; negative = better for error metrics

| Metric | SHORT | MEDIUM | LONG |
|--------|-------|--------|------|
| Desc. Accuracy | +0.02 | — | +0.79 |
| Pos. Error (m) | -159.81 | -151.11 | -147.79 |
| Temporal Error (s) | -7.11 | -1.45 | -9.39 |
