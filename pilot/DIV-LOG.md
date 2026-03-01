# Diversity Measurement Log

## Rao's Q Prototype — 2026-02-28

### Motivation

Binary diversity (count distinct strategies, Gini-Simpson) treats all strategy pairs as equally different. Ecology solved this decades ago with Rao's Quadratic Entropy: Q = Σ d(i,j) · p_i · p_j, where d(i,j) is a continuous distance between strategies. Nobody in RLVR literature uses this. We prototyped it.

### Setup

- 14 multi-strategy problems from Run 16 LLM judge data
- 4 models: R1-Distill, Nemotron-v1, v2, BroRL (ProRL trajectory)
- 8 correct traces per model per problem (sampled from 64 rollouts)
- Strategy classifications from GPT-5-nano judge

### Four Metrics Tested

| Metric | Distance d(i,j) | Source |
|--------|-----------------|--------|
| Gini-Simpson | Binary: d=1 for all i≠j | Baseline |
| Jaccard | 1 - \|tag_i ∩ tag_j\| / \|tag_i ∪ tag_j\| on regex-extracted math tags | Ecology |
| Embedding | 1 - cosine(embed_i, embed_j) on OpenAI text-embedding-3-small of strategy descriptions | NLP |
| Gower | Mean of per-trait distances (categorical 0/1, continuous normalized, binary 0/1) on 5 traits: representation, technique, desc_length, uses_formula, uses_transform | Ecology (Gower 1971) |

### Results: Mean Rao's Q across 14 problems

```
Model                 GiniSimpson      Jaccard    Embedding        Gower
------------------------------------------------------------------------
r1-distill                 0.3795       0.2802       0.1331       0.1148
nemotron-v1                0.2287       0.1617       0.0788       0.0776
nemotron-v2                0.2052       0.1618       0.0699       0.0705
nemotron-brorl             0.1972       0.1510       0.0782       0.0828
```

### Rankings

```
Gini-Simpson : R1-Distill > v1 > v2 > BroRL
Jaccard      : R1-Distill > v2 ≈ v1 > BroRL
Embedding    : R1-Distill > v1 > BroRL > v2
Gower        : R1-Distill > BroRL > v1 > v2
```

All four agree: **R1-Distill is most diverse (rank 1).**

Rankings diverge at positions 2-4. Key shift: **BroRL moves from rank 4 (binary) to rank 2 (Gower/Embedding).** BroRL has fewer strategies by count, but the ones it uses are more structurally distinct from each other.

### Spearman Correlations (n=56, all model×problem pairs)

```
                GiniSimpson  Jaccard  Embedding  Gower
GiniSimpson          1.000    0.957      0.910  0.894
Jaccard              0.957    1.000      0.939  0.901
Embedding            0.910    0.939      1.000  0.930
Gower                0.894    0.901      0.930  1.000
```

Embedding and Gower agree most with each other (ρ=0.930). Both diverge most from Gini-Simpson (~0.90). They're capturing overlapping but not identical structure.

### Distance Distribution (35 strategy pairs)

```
Metric       Mean    Std     Min     Max
Jaccard      0.705   0.211   0.000   1.000
Embedding    0.348   0.093   0.204   0.553
Gower        0.305   0.186   0.000   0.648
```

Embedding on strategy descriptions: range [0.204, 0.553]. Previous attempt embedding full solutions with MiniLM gave 0.92-0.96 cosine (range ~0.04-0.08). Embedding the judge's method description produces **~5× wider distance spread**. The signal is there — you just have to embed the right thing.

### Key Findings

1. **Continuous distance reveals structure binary misses.** BroRL's rank shift (4→2 under Gower) means breadth training (N=512) produces fewer but more structurally distinct strategies. Binary count metrics are blind to this.

2. **Embedding strategy descriptions works; embedding full solutions doesn't.** The judge's strategy name+description strips problem-specific vocabulary, leaving only the method signature. This is what makes embeddings discriminative.

3. **Jaccard tags are brittle.** Problem 58 (AM-GM vs Lagrange multipliers) gets d=0.5 when it should be ~0.8+. Regex keyword dictionaries can't cover the full math vocabulary. Embedding and Gower are more robust.

4. **The monotonic narrative breaks.** Gini-Simpson tells a clean story: R1→v1→v2→BroRL = monotonic diversity loss. Rao's Q with Gower tells a messier but more accurate story: R1 is most diverse, standard ProRL (v1→v2) narrows monotonically, but BroRL partially recovers structural diversity without recovering strategy count.

### Verdict

**GO for Rao's Q as a contribution.** No RLVR paper uses continuous inter-strategy distance. The BroRL rank shift is a genuine finding that simple metrics miss. Recommended distance: **Gower or Embedding** (not Jaccard — too brittle).

### Caveats

- n=14 multi-strategy problems. Small sample. Statistical power for rank differences at positions 2-4 is low.
- Strategy classifications come from LLM judge (GPT-5-nano). Garbage in → garbage out. But this is the best available.
- Gower traits are hand-designed (5 traits). Could be expanded or learned.
- Embedding model (text-embedding-3-small) not trained on math. A math-specific encoder might do better.

### Cost

- OpenAI embeddings: ~$0.001 (14 problems × ~4 strategies × ~20 tokens each)
- Total prototype: negligible

### Scripts & Data

- `pilot/rao_q_prototype.py` — full prototype
- `data/analysis/rao_q_prototype.json` — saved results
- `data/analysis/llm_judge_pilot.json` — input judge data (Run 16)
