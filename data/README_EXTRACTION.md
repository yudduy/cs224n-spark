# Data Extraction Summary: Explorability vs. Diversity

## Files Generated

### 1. **PAPER_SYNTHESIS.md** (17 KB)
Comprehensive structured summary for paper writing. Contains:

**Pass@K Confidence Intervals** (Section 1)
- Easy, Medium, Hard, All tiers
- All models: R1-Distill, Nemotron v1, v2, BroRL
- All k values: k=1,2,4,8,16,32,64
- 95% confidence intervals

**Wilcoxon Test Results** (Section 2)
- V1 anomaly: Hard-tier collapse (R1 vs V1)
- V2 vs R1: Pairwise comparisons across tiers
- P-values, significance markers (*, **, ***)
- Per-problem win counts

**Sampling Efficiency** (Section 3)
- Cross-model k-matching (how many samples to equalize pass@1)
- Easy, Medium, Hard tiers
- All 16 model pairs

**Coverage Analysis** (Section 4)
- Problems solved/unsolved per tier
- Exclusive wins (R1-only, V2-only)
- Hard tier expansion: 4 problems solved only by V2 (42, 44, 50, 51)

**Strategy Diversity** (Section 5)
- 60 problems analyzed, 14 with multiple strategies (23%)
- Per-tier breakdown: 5, 4, 5 (easy, medium, hard)
- 14 multi-strategy problems fully detailed with per-model correct counts
- ALL 14 show V2 ≥ R1 (no narrowing cases)
- Largest gains: problem 35 (+51pp), 19 (+45pp), 58 (+33pp)

**Key Patterns & Interpretations** (Section 6)
- V1 anomaly explained (narrowness ≠ efficiency)
- V2 expansion beyond coverage
- V1 ≠ rejection sampler (hard tier disproof)
- Confidence gradient (where LLM judge uncertain, V2 gains largest)
- Pass@k saturation patterns by tier

**Paper Narrative** (Section 7)
- "The Death of the Rejection Sampler Hypothesis"
- 3-part evidence structure:
  1. Hard tier coverage loss (non-monotonic improvement)
  2. Sampling efficiency (anti-efficiency of V1)
  3. Multi-strategy gains (restructuring, not re-weighting)

**Statistical Metadata** (Section 10)
- 3,840 total inference runs (60 problems × 64 rollouts × 4 models)
- Binomial CI, Wilcoxon test methodology
- No multiple comparison correction (exploratory)

---

### 2. **EXTRACTED_KEY_METRICS.json** (4.8 KB)
Structured data export for programmatic access:

**Structure**:
```json
{
  "metadata": {...},
  "pass_at_k_summary": {
    "easy": {...},
    "medium": {...},
    "hard": {...},
    "all": {...}
  },
  "v1_anomaly": {...},
  "sampling_efficiency": {...},
  "coverage": {...},
  "strategy_diversity": {...},
  "key_findings": [...]
}
```

**Use cases**:
- Building tables for paper/slides
- Programmatic comparison of models
- Verifying citations in appendix
- Generating figures (pass@k curves, efficiency heatmaps)

**Key nested data**:
- All pass@k values (mean, CI lo/hi)
- All Wilcoxon results (meandiff, p-value, n_better)
- Problem-level coverage (by ID)
- Multi-strategy problems (by ID) with strategy counts
- V2 expansion cases (13/14) and gains

---

### 3. **NARRATIVE_ARC.md** (9.4 KB)
Conceptual roadmap for paper argument. Contains:

**Sections**:
1. Conventional wisdom (rejection sampler hypothesis as straw)
2. First crack (hard tier anomaly)
3. Smoking gun (sampling efficiency paradox)
4. Breakthrough (V2 recovery)
5. Evidence of explorability (multi-strategy gains)
6. Narration (theoretical implications)
7. Remaining questions (open research directions)
8. Death knell (three pieces of evidence)
9. Paper structure (narrative arc blueprint)
10. Quotes for motivation (pull-quotes for introduction/discussion)
11. Supporting tables (appendix references)
12. Kill test experiment (how to resolve the reframe)

**Narrative flow**:
- Setup → Clash → Crisis → Turning Point → Resolution → Implication
- Each section tied to specific data points from PAPER_SYNTHESIS.md

**Ready-to-use elements**:
- Discussion structure ("The Death of the Rejection Sampler")
- Motivation quotes
- Experiment design for next work
- Appendix navigation

---

## Quick Reference: Key Data Points

### Pass@K (Overall)
| Metric | Value | Source |
|--------|-------|--------|
| R1→V2 gain at pass@1 | +13.2pp | EXTRACTED_KEY_METRICS.json, all tier |
| V2 vs R1 pvalue at easy k=1 | <0.001 | PAPER_SYNTHESIS.md §2.2 |
| V2 vs R1 pvalue at hard k=1 | 0.0064 | PAPER_SYNTHESIS.md §2.2 |
| V2 vs R1 hard k=8 gain | +17.5pp | PAPER_SYNTHESIS.md §2.2 |

### V1 Anomaly
| Metric | Value | Source |
|--------|-------|--------|
| Hard k=8 collapse (V1 vs R1) | -10.4pp | EXTRACTED_KEY_METRICS.json |
| Hard k=8 pvalue | 0.0049 | PAPER_SYNTHESIS.md §2.1 |
| Problems favoring R1 on hard k=8 | 9/20 | PAPER_SYNTHESIS.md §2.1 |
| Medium tier anomaly | None (ns) | PAPER_SYNTHESIS.md §2.1 |

### Sampling Efficiency (Hard Tier)
| Comparison | K Needed |
|------------|----------|
| V1→V2 (to match V2's pass@1) | 9 |
| R1→V2 (to match V2's pass@1) | 3 |
| V2→R1 (to match R1's pass@1) | 9 |

**Interpretation**: V1 is 9x LESS efficient than V2 on hard tier. Rejection sampler hypothesis would predict V1 ≥ R1 in efficiency. ✗

### Coverage (Hard Tier)
- V2-only (new hard problems): 4 (42, 44, 50, 51)
- R1-only: 1 (57)
- Both solve: 14/20

### Multi-Strategy Problems (14 Total)
- V2 expansion (V2 > R1): 13/14 (93%)
- V2 parity (V2 = R1): 1/14 (7%)
- V2 narrowing (V2 < R1): 0/14 (0%)

Largest gains:
- Problem 35 (4 strategies): +51pp
- Problem 19 (3 strategies): +45pp
- Problem 58 (2 strategies): +33pp

---

## How to Use These Files

### For Paper Writing
1. Start with **NARRATIVE_ARC.md** to understand the argument structure
2. Use **PAPER_SYNTHESIS.md** §1-5 for all tables/data points
3. Copy confidence intervals and Wilcoxon results directly into main text
4. Pull pull-quotes from NARRATIVE_ARC.md §10 for introduction

### For Appendix
- **PAPER_SYNTHESIS.md** §8: Summary table (1-pager)
- **PAPER_SYNTHESIS.md** §1-5: Full tables (pass@k, Wilcoxon, sampling efficiency, coverage, strategy data)
- **PAPER_SYNTHESIS.md** §10: Statistical metadata

### For Figures
- Use EXTRACTED_KEY_METRICS.json to generate:
  - Pass@k curves (by tier, by model)
  - Sampling efficiency heatmap (k_needed as color)
  - Strategy gain scatter plot (n_strategies vs. V2 gain)
  - Coverage Venn diagram (hard tier: R1, V2, both solve)

### For Presentation
- **NARRATIVE_ARC.md** §1-9: Slide outline (12 slides)
- EXTRACTED_KEY_METRICS.json: Data points for speaker notes
- PAPER_SYNTHESIS.md §7: "The Death of the Rejection Sampler" as main narrative

---

## Data Quality Notes

### Confidence Intervals
- Method: Binomial (exact), 95% confidence
- All pass@k values ≤64 (bounded correctly)
- CIs tighten as k increases (expected)

### Wilcoxon Tests
- Paired non-parametric test on per-problem pass@k
- P-values: exact where computed, NaN where all problems tied
- Multiple comparison: none applied (exploratory analysis)
- Significance markers: * p<0.05, ** p<0.01, *** p<0.001

### Strategy Diversity
- 60 problems analyzed by LLM judge (Claude 3.5 Sonnet)
- Confidence levels: low (5), medium (16), high (39)
- Multi-strategy: 14 problems, 2-4 strategies each
- Per-model correct: /64 rollouts

### Coverage
- "Solves" = pass@64 > 0 (at least 1/64 correct)
- R1-only/V2-only: mutual exclusivity verified
- Hard tier expansion: 4 new problems (42, 44, 50, 51)

---

## Files Generated (Summary)

```
/Users/duy/Documents/build/cs224n-spark/data/
├── PAPER_SYNTHESIS.md (17 KB) ← Main reference
├── EXTRACTED_KEY_METRICS.json (4.8 KB) ← Programmatic access
├── NARRATIVE_ARC.md (9.4 KB) ← Conceptual roadmap
└── README_EXTRACTION.md (this file)
```

**Total**: 31 KB of structured, paper-ready data.

---

## Next Steps

1. **Paper outline**: Follow NARRATIVE_ARC.md §9 (paper structure)
2. **Results section**: Copy tables from PAPER_SYNTHESIS.md §1-5
3. **Discussion**: Use PAPER_SYNTHESIS.md §6-7 and NARRATIVE_ARC.md §6-8
4. **Figures**: Generate from EXTRACTED_KEY_METRICS.json
5. **Appendix**: Full tables from PAPER_SYNTHESIS.md

All data is citation-ready with source file references.

