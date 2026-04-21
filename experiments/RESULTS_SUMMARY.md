# ReMEmbR v2 Ablation Study — Results Summary
Date: 2026-04-20

## Setup
- Sequences: seq0, seq4
- Pipelines: v1 baseline, CLIP+VLM (t95/t90), CLIP-only (t95/t90), Random (t95/t90)
- LLM: GPT-4o (temperature=0)
- Questions per sequence: 30 (all answered, zero API failures)
- Scoring thresholds: position ≤15m, time ≤0.5min, duration ≤0.5min, binary exact match

---

## TABLE 1: Overall Accuracy (scoreable question types only)

| Pipeline             | Overall    | time       | duration   | position   | binary     |
|----------------------|------------|------------|------------|------------|------------|
| v1_seq0              | 50.0% (15/30) | 85.7% (6/7) | 50.0% (1/2) | 41.7% (5/12) | 75.0% (3/4) |
| v2_seq0_clipvlm_t95  | **73.3% (22/30)** | 100.0% (7/7) | 100.0% (2/2) | 75.0% (9/12) | 100.0% (4/4) |
| v2_seq0_clipvlm_t90  | 53.3% (16/30) | 85.7% (6/7) | 100.0% (2/2) | 50.0% (6/12) | 50.0% (2/4) |
| v2_seq0_clip_t95     | 53.3% (16/30) | 100.0% (7/7) | 50.0% (1/2) | 41.7% (5/12) | 75.0% (3/4) |
| v2_seq0_clip_t90     | 43.3% (13/30) | 100.0% (7/7) | 50.0% (1/2) | 25.0% (3/12) | 50.0% (2/4) |
| v2_seq0_random_t95   | 53.3% (16/30) | 85.7% (6/7) | 50.0% (1/2) | 58.3% (7/12) | 50.0% (2/4) |
| v2_seq0_random_t90   | 43.3% (13/30) | 71.4% (5/7) | 50.0% (1/2) | 50.0% (6/12) | 25.0% (1/4) |
| v1_seq4              | 36.7% (11/30) | 66.7% (2/3) | 0.0% (0/1) | 70.0% (7/10) | 25.0% (2/8) |
| v2_seq4_clipvlm_t95  | 43.3% (13/30) | 66.7% (2/3) | 0.0% (0/1) | 50.0% (5/10) | 75.0% (6/8) |
| v2_seq4_clipvlm_t90  | 40.0% (12/30) | 66.7% (2/3) | 0.0% (0/1) | 50.0% (5/10) | 62.5% (5/8) |
| v2_seq4_clip_t95     | **56.7% (17/30)** | 66.7% (2/3) | 0.0% (0/1) | 80.0% (8/10) | 87.5% (7/8) |
| v2_seq4_clip_t90     | 33.3% (10/30) | 33.3% (1/3) | 0.0% (0/1) | 50.0% (5/10) | 50.0% (4/8) |
| v2_seq4_random_t95   | 43.3% (13/30) | 33.3% (1/3) | 100.0% (1/1) | 70.0% (7/10) | 50.0% (4/8) |
| v2_seq4_random_t90   | 36.7% (11/30) | 33.3% (1/3) | 100.0% (1/1) | 50.0% (5/10) | 50.0% (4/8) |

Note: 5 questions in seq0 and 8 in seq4 are type=text (no numeric metric) — scored separately below.

---

## TABLE 2: Accuracy by Length Category

| Pipeline             | LONG          | MEDIUM        | SHORT         |
|----------------------|---------------|---------------|---------------|
| v1_seq0              | 40.0% (4/10)  | 40.0% (4/10)  | 70.0% (7/10)  |
| v2_seq0_clipvlm_t95  | **90.0% (9/10)** | 60.0% (6/10) | 70.0% (7/10) |
| v2_seq0_clipvlm_t90  | 80.0% (8/10)  | 30.0% (3/10)  | 50.0% (5/10)  |
| v2_seq0_clip_t95     | 40.0% (4/10)  | 50.0% (5/10)  | 70.0% (7/10)  |
| v2_seq0_clip_t90     | 50.0% (5/10)  | 40.0% (4/10)  | 40.0% (4/10)  |
| v2_seq0_random_t95   | 60.0% (6/10)  | 40.0% (4/10)  | 60.0% (6/10)  |
| v2_seq0_random_t90   | 40.0% (4/10)  | 40.0% (4/10)  | 50.0% (5/10)  |
| v1_seq4              | 30.0% (3/10)  | 50.0% (5/10)  | 30.0% (3/10)  |
| v2_seq4_clipvlm_t95  | 20.0% (2/10)  | 60.0% (6/10)  | 50.0% (5/10)  |
| v2_seq4_clipvlm_t90  | 30.0% (3/10)  | 50.0% (5/10)  | 40.0% (4/10)  |
| v2_seq4_clip_t95     | 50.0% (5/10)  | **60.0% (6/10)** | **60.0% (6/10)** |
| v2_seq4_clip_t90     | 20.0% (2/10)  | 60.0% (6/10)  | 20.0% (2/10)  |
| v2_seq4_random_t95   | 50.0% (5/10)  | 60.0% (6/10)  | 20.0% (2/10)  |
| v2_seq4_random_t90   | 30.0% (3/10)  | 50.0% (5/10)  | 30.0% (3/10)  |

---

## TABLE 3: Seq0 Per-Type × Category Breakdown

### time (7 questions)
| Pipeline             | LONG        | MEDIUM      | SHORT       | ALL         |
|----------------------|-------------|-------------|-------------|-------------|
| v1_seq0              | 100% (2/2)  | 66.7% (2/3) | 100% (2/2)  | 85.7% (6/7) |
| v2_seq0_clipvlm_t95  | 100% (2/2)  | 100% (3/3)  | 100% (2/2)  | 100% (7/7)  |
| v2_seq0_clipvlm_t90  | 100% (2/2)  | 66.7% (2/3) | 100% (2/2)  | 85.7% (6/7) |
| v2_seq0_clip_t95     | 100% (2/2)  | 100% (3/3)  | 100% (2/2)  | 100% (7/7)  |
| v2_seq0_clip_t90     | 100% (2/2)  | 100% (3/3)  | 100% (2/2)  | 100% (7/7)  |
| v2_seq0_random_t95   | 100% (2/2)  | 66.7% (2/3) | 100% (2/2)  | 85.7% (6/7) |
| v2_seq0_random_t90   | 100% (2/2)  | 33.3% (1/3) | 100% (2/2)  | 71.4% (5/7) |

### duration (2 questions)
| Pipeline             | LONG        | SHORT       | ALL         |
|----------------------|-------------|-------------|-------------|
| v1_seq0              | 0% (0/1)    | 100% (1/1)  | 50% (1/2)   |
| v2_seq0_clipvlm_t95  | 100% (1/1)  | 100% (1/1)  | 100% (2/2)  |
| v2_seq0_clipvlm_t90  | 100% (1/1)  | 100% (1/1)  | 100% (2/2)  |
| v2_seq0_clip_t95     | 0% (0/1)    | 100% (1/1)  | 50% (1/2)   |
| v2_seq0_clip_t90     | 0% (0/1)    | 100% (1/1)  | 50% (1/2)   |
| v2_seq0_random_t95   | 0% (0/1)    | 100% (1/1)  | 50% (1/2)   |
| v2_seq0_random_t90   | 0% (0/1)    | 100% (1/1)  | 50% (1/2)   |

### position (12 questions)
| Pipeline             | LONG        | MEDIUM      | SHORT       | ALL          |
|----------------------|-------------|-------------|-------------|--------------|
| v1_seq0              | 16.7% (1/6) | 50% (2/4)   | 100% (2/2)  | 41.7% (5/12) |
| v2_seq0_clipvlm_t95  | 83.3% (5/6) | 75% (3/4)   | 50% (1/2)   | 75.0% (9/12) |
| v2_seq0_clipvlm_t90  | 66.7% (4/6) | 25% (1/4)   | 50% (1/2)   | 50.0% (6/12) |
| v2_seq0_clip_t95     | 16.7% (1/6) | 50% (2/4)   | 100% (2/2)  | 41.7% (5/12) |
| v2_seq0_clip_t90     | 33.3% (2/6) | 25% (1/4)   | 0% (0/2)    | 25.0% (3/12) |
| v2_seq0_random_t95   | 50% (3/6)   | 50% (2/4)   | 100% (2/2)  | 58.3% (7/12) |
| v2_seq0_random_t90   | 33.3% (2/6) | 75% (3/4)   | 50% (1/2)   | 50.0% (6/12) |

### binary (4 questions)
| Pipeline             | LONG        | SHORT       | ALL         |
|----------------------|-------------|-------------|-------------|
| v1_seq0              | 100% (1/1)  | 66.7% (2/3) | 75% (3/4)   |
| v2_seq0_clipvlm_t95  | 100% (1/1)  | 100% (3/3)  | 100% (4/4)  |
| v2_seq0_clipvlm_t90  | 100% (1/1)  | 33.3% (1/3) | 50% (2/4)   |
| v2_seq0_clip_t95     | 100% (1/1)  | 66.7% (2/3) | 75% (3/4)   |
| v2_seq0_clip_t90     | 100% (1/1)  | 33.3% (1/3) | 50% (2/4)   |
| v2_seq0_random_t95   | 100% (1/1)  | 33.3% (1/3) | 50% (2/4)   |
| v2_seq0_random_t90   | 0% (0/1)    | 33.3% (1/3) | 25% (1/4)   |

---

## TABLE 4: Seq4 Per-Type × Category Breakdown

### time (3 questions)
| Pipeline             | SHORT       | ALL         |
|----------------------|-------------|-------------|
| v1_seq4              | 66.7% (2/3) | 66.7% (2/3) |
| v2_seq4_clipvlm_t95  | 66.7% (2/3) | 66.7% (2/3) |
| v2_seq4_clipvlm_t90  | 66.7% (2/3) | 66.7% (2/3) |
| v2_seq4_clip_t95     | 66.7% (2/3) | 66.7% (2/3) |
| v2_seq4_clip_t90     | 33.3% (1/3) | 33.3% (1/3) |
| v2_seq4_random_t95   | 33.3% (1/3) | 33.3% (1/3) |
| v2_seq4_random_t90   | 33.3% (1/3) | 33.3% (1/3) |

### position (10 questions)
| Pipeline             | LONG        | MEDIUM       | SHORT       | ALL          |
|----------------------|-------------|--------------|-------------|--------------|
| v1_seq4              | 50% (3/6)   | 100% (3/3)   | 100% (1/1)  | 70% (7/10)   |
| v2_seq4_clipvlm_t95  | 16.7% (1/6) | 100% (3/3)   | 100% (1/1)  | 50% (5/10)   |
| v2_seq4_clipvlm_t90  | 33.3% (2/6) | 66.7% (2/3)  | 100% (1/1)  | 50% (5/10)   |
| v2_seq4_clip_t95     | 66.7% (4/6) | 100% (3/3)   | 100% (1/1)  | **80% (8/10)** |
| v2_seq4_clip_t90     | 16.7% (1/6) | 100% (3/3)   | 100% (1/1)  | 50% (5/10)   |
| v2_seq4_random_t95   | 50% (3/6)   | 100% (3/3)   | 100% (1/1)  | 70% (7/10)   |
| v2_seq4_random_t90   | 16.7% (1/6) | 100% (3/3)   | 100% (1/1)  | 50% (5/10)   |

### binary (8 questions)
| Pipeline             | LONG        | MEDIUM      | SHORT       | ALL          |
|----------------------|-------------|-------------|-------------|--------------|
| v1_seq4              | 0% (0/1)    | 50% (2/4)   | 0% (0/3)    | 25% (2/8)    |
| v2_seq4_clipvlm_t95  | 100% (1/1)  | 75% (3/4)   | 66.7% (2/3) | 75% (6/8)    |
| v2_seq4_clipvlm_t90  | 100% (1/1)  | 75% (3/4)   | 33.3% (1/3) | 62.5% (5/8)  |
| v2_seq4_clip_t95     | 100% (1/1)  | 75% (3/4)   | 100% (3/3)  | **87.5% (7/8)** |
| v2_seq4_clip_t90     | 100% (1/1)  | 75% (3/4)   | 0% (0/3)    | 50% (4/8)    |
| v2_seq4_random_t95   | 100% (1/1)  | 75% (3/4)   | 0% (0/3)    | 50% (4/8)    |
| v2_seq4_random_t90   | 100% (1/1)  | 50% (2/4)   | 33.3% (1/3) | 50% (4/8)    |

---

## TABLE 5: Text Questions (manual scoring, not in numeric eval)

### SEQ0 (5 text questions)
Ground truths: blue puffer jacket | right sidewalk side | red couches | left street side | turned left

| Question                      | v1  | clipvlm_t95 | clipvlm_t90 | clip_t95 | clip_t90 | random_t95 | random_t90 |
|-------------------------------|-----|-------------|-------------|----------|----------|------------|------------|
| jacket color (blue puffer)    | ✓   | ✓           | ✓           | ✓        | ✓        | ✓          | ✓          |
| sidewalk side (right)         | ✗   | ✓           | ✓           | ✓        | ✓        | ✗          | ✓          |
| couch color (red)             | ✓   | ✓           | ✓           | ✓        | ✓        | ✗          | ✓          |
| street side (left)            | ✗   | ✗           | ✗           | ✗        | ✗        | ✗          | ✗          |
| turn direction (left)         | ✗   | ✗           | ✗           | ✗        | ✗        | ✗          | ✗          |
| **TOTAL**                     | **2/5** | **3/5** | **3/5** | **3/5** | **3/5** | **1/5** | **3/5** |

Notes:
- "Street side (left)" wrong across all — robot navigates a pedestrian pathway, doesn't know which street side
- "Turn direction (left)" wrong across all — trajectory data points east/right, ground truth says left
- v1 missed sidewalk side; random_t95 got the fewest (1/5)

### SEQ4 (8 text questions; v1 baseline had no text responses)
Ground truths: beige hat | 2 turns before building | white car | no cars at gate | 2 doors | 2 turns (short) | 3 turnarounds | turned right

| Question                      | clipvlm_t95 | clipvlm_t90 | clip_t95 | clip_t90 | random_t95 | random_t90 |
|-------------------------------|-------------|-------------|----------|----------|------------|------------|
| hat color (beige)             | ✗           | ✗           | ✗        | ✗        | ✗          | ✗          |
| turns before building (2)     | ✗           | ✓           | ✗        | ✗        | ✗          | ✓          |
| car color (white)             | ✗           | ✗           | ✗        | ✗        | ✗          | ✗          |
| cars at gate (none)           | ✓           | ✓           | ✗        | ✓        | ✓          | ✓          |
| doors opened (2)              | ✗           | ✓           | ✓        | ✗        | ✓          | ✓          |
| turns short (2)               | ✗           | ✗           | ✗        | ✗        | ✗          | ✗          |
| turnarounds (3)               | ✗           | ✗           | ✗        | ✗        | ✗          | ✗          |
| turn right                    | ✗           | ✗           | ✗        | ✗        | ✗          | ✓          |
| **TOTAL**                     | **1/8**     | **3/8**     | **1/8**  | **1/8**  | **2/8**    | **4/8**    |

Notes:
- Hat color (beige) and car color (white) wrong across all — visual detail not captured in memory
- Turnaround count (3) wrong across all — agents consistently say robot did not turn around
- random_t90 gets most (4/8) — partially luck on turn_right and cars_at_gate
- v1_seq4 had no text-type responses in the output files (0/8)

---

## Key Takeaways

1. **Best overall on seq0: v2_clipvlm_t95 at 73.3%** — +23pp over v1 (50%). Dominant on LONG questions (90% vs 40%).
2. **Best overall on seq4: v2_clip_t95 at 56.7%** — +20pp over v1 (36.7%). Best position (80%) and binary (87.5%).
3. **t95 consistently beats t90** across both sequences and both retrieval methods.
4. **CLIP+VLM wins on seq0** — longer sequences benefit from VLM-generated descriptions enabling better semantic retrieval.
5. **CLIP-only wins on seq4** — seq4 appears to have queries that match visual embeddings directly without needing language descriptions.
6. **Random baseline is surprisingly competitive on seq4 position (70%)** — matching v1 — suggesting the agent's reasoning from keyframes matters more than retrieval quality for some question types.
7. **Text questions are hard overall** — "street side" and "turn direction" (seq0) and "hat color", "car count", "turnarounds" (seq4) are universally wrong across all pipelines, suggesting memory granularity limits for fine visual details and exact motion counting.
