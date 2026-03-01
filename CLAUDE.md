# CS224N Spark — Project Instructions

## Research Context
CS224N custom project. Student: Duy Nguyen. Professor: Yejin Choi (co-author ProRL, BroRL). TA Mentor: Mirac Suzgun.

## Intellectual Foundation (from distillation)

### The Root
Learning is lossy compression. Any finite-capacity system updating toward observed reward must shed representation of unobserved alternatives. Entropy reduction under constraints (Jaynes). The tension: learning anything at all vs. retaining the ability to learn something different. 150-year-old question — Boltzmann, Darwin, Holland. LLMs are the latest substrate.

### First Principles

**Foundations (hard to vary — negate and the answer breaks):**
1. **⬛ Optimization compresses support.** p(a) ~ exp(R(a)/T); training contracts support. [Boltzmann; pass@1↑ pass@k↓ in every RLVR system]
2. **⬛ Coverage costs energy.** No system maintains diversity for free. Entropy bonuses, diversity objectives, or population methods — always a tax. [Second law; confirmed: GANs, RL, evolution]
3. **⬛ The KL leash bounds reachability.** Novel strategies outside the reference policy's support are unreachable. [ProRL 2505.24864; Invisible Leash 2507.14843]

**Open (where discoveries hide — competing framings unresolved):**
4. **⬜ The objective is the bottleneck.** KL-regularized expected-return maximization specifies unimodal target. Optimizer works perfectly — optimizes the wrong thing. [2510.20817; 2601.21669] *Varies:* does objective redesign suffice, or must we abandon objectives entirely? (Stanley's novelty search vs. better-specified rewards)
5. **⬜ Exploration must match landscape structure.** Reasoning is high-K (rugged). But landscape metaphor breaks for discrete token generation. [Kauffman NK 1987; NFL 1997] *Varies:* do continuous landscape intuitions transfer to autoregressive LLMs?
6. **⬜ Novel strategies emerge adjacently.** Adjacent possible (Kauffman). But recombination can create distant jumps. [ProRL expanding base-model support] *Varies:* boundary extension vs. combinatorial recombination as the mechanism.
7. **⬜ Explore proportionally to uncertainty, not uniformly.** PUCT allocates to undervisited branches. Fork tokens (20% of positions) drive all learning. [Lai & Robbins 1985; Beyond 80/20 2506.01939] *Varies:* does token-level uncertainty-proportional exploration translate to strategy-level diversity?

### Key Insights
1. Objective designs the collapse — KL-regularized RL specifies unimodal target [2510.20817]
2. Expected return itself is the culprit — exponential log-prob divergence, fix: IPS [2601.21669]
3. Token entropy ↑ while answer diversity ↓ — surface metrics misleading [Invisible Leash 2507.14843]
4. Strategy collapse is a Pólya urn — early wins compound via sample-weighted updates [Pólya 1930]
5. Reference-policy reset = re-annealing — ProRL widens adjacent possible [2505.24864; Boltzmann]
6. Fork tokens are where diversity lives and dies — 20% of positions drive all RLVR learning [2506.01939]
7. Go-Explore failure modes apply — detachment + derailment in RLVR [Ecoffet Nature 2021]
8. Metadynamics is strongest cross-disciplinary analog — convergence guarantees RLVR lacks [Laio 2002]
9. Measurement problem is hidden bottleneck — RPD achieves 53%, barely above random [2510.26122]
10. **Diversity is the wrong target; explorability is the right one** — TTT-Discover proves inference-time discovery works [2601.16175]

### Contrarian Truth
The field has causality backwards. Dominant narrative: "RLVR causes collapse; fix the optimizer." Evidence: the objective SPECIFIES collapse. Every optimizer fix (ACE, MARA, diversity bonuses) compensates for an objective-level flaw.

Inference-time exploration may be strictly superior to training-time diversity. If you can explore at test time, you don't need to STORE diversity — you need to store EXPLORABILITY. The ability to explore ≠ the state of being diverse.

### The Next Question (⬜ P4 × Insight 10)
Can we optimize for EXPLORABILITY rather than diversity or expected return?

**Resolving experiment:** Compare (i) diversity-preserving training + fixed inference vs. (ii) standard RLVR + equal-cost inference-time search. Kill-test methodology is the natural tool.

**Our feasibility-constrained proxy:** Can't train (i) ourselves. Instead: ProRL trajectory = natural experiment. R1-Distill → Nemotron v1 → v2 → BroRL = 4 checkpoints, increasing RL investment. Question: does more training create explorability, or just convergence?

## Current Experiment

### Design
4 models × 60 MATH problems (20 easy/20 med/20 hard) × 64 rollouts. Strategy-level analysis.

### Models (verified HF MCP 2026-02-27)
All 1.5B Qwen2. Nemotron finetuned FROM R1-Distill = same trajectory.

| Checkpoint | Training | MATH | AIME24 | HuggingFace |
|---|---|---|---|---|
| R1-Distill | Distilled from R1 671B | 82.90 | 28.54 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Nemotron v1 | ProRL 2K steps | 91.89 | 48.13 | `nvidia/Nemotron-...-1.5B` rev `v1` |
| Nemotron v2 | ProRL 3K steps | 92.49 | 49.58 | `nvidia/Nemotron-...-1.5B` default |
| Nemotron BroRL | v2 + 419 steps N=512 | 92.20 | 60.42 | `nvidia/Nemotron-...-1.5B` branch `brorl` |

### HF Data (no 1.5B traces exist — must generate ~$1-3)
- `nvidia/OpenMathReasoning` — R1 671B, ~10-16 CoT/problem → strategy oracle
- `uzaymacar/math-rollouts` — R1-Distill 8B/14B, chunk annotations → annotation schema (function tags: problem_setup, fact_retrieval, active_computation, plan_generation, uncertainty_management)
- `open-r1/OpenR1-Math-Raw` — R1 671B, 1-8 rollouts → reference traces

### Decision Tree
| Finding | Implication |
|---------|-------------|
| R1-Distill+search ≈ Nemotron+search | Extra RL is wasted with search budget |
| Nemotron+search >> R1-Distill+search | ProRL creates genuine explorability |
| BroRL finds strategies absent in v1/v2 | Breadth (N=512) discovers what depth misses |
| Results vary by difficulty | Phase transition — strongest paper |
| Strategy count drops R1-Distill → Nemotron | ProRL narrows strategies (cost of convergence) |
| Strategy count rises R1-Distill → Nemotron | ProRL expands strategy space (contradicts "rejection sampler") |

## Source Hierarchy (by Lindy-ness)

### Timeless (100+ years)
- Boltzmann/Kirkpatrick (1983) — Simulated annealing, acceptance probability
- Darwin/Fisher (1930)/Wright (1932) — Selection vs drift, shifting balance
- Pólya (1930) — Urn models, preferential attachment
- Shannon (1948) — Entropy as diversity measure
- Jaynes (1957) — Maximum entropy principle

### Enduring (10-100 years)
- Holland (1975) — Schema theorem, building blocks
- Kauffman (1987, 2000) — NK landscapes, adjacent possible
- Wolpert & Macready (1997) — No free lunch
- Lai & Robbins (1985) — UCB, bandit exploration
- Stanley & Lehman (2011) — Novelty search on deceptive landscapes
- Mouret & Clune (2015) — MAP-Elites, quality-diversity
- Sutton et al. (1999) — Options framework
- Schmidhuber (2010) — Compression progress as curiosity
- Bellemare et al. (2016) — Count-based exploration, pseudo-counts
- Ecoffet et al. (2021) — Go-Explore, detachment/derailment
- Laio & Parrinello (2002) — Metadynamics

### Recent (RLVR-specific)
- Yue et al. NeurIPS 2025 — RLVR is rejection sampler, 2504.13837
- Invisible Leash 2025 — Token entropy ↑ answer diversity ↓, 2507.14843
- ProRL 2025 — Reference policy reset, 2505.24864
- BroRL 2025 — Breadth scaling, mass balance, 2510.01180
- TTT-Discover 2026 — Test-time PUCT exploration, 2601.16175
- Rewarding the Rare 2026 — Strategy-level exploration, 2601.08763
- KL-RL Mode Collapse 2025 — Objective designs collapse, 2510.20817
- Expected Return Collapse 2026 — IPS fix, 2601.21669
- Beyond 80/20 2025 — Fork tokens drive learning, 2506.01939
- DPH-RL 2025 — Divergence choice matters, 2509.07430
- DAPO 2025 — Clip-Higher entropy management, 2503.14476
- VCRL 2025 — Variance-based curriculum, 2509.19803

## Verification Tools — ALWAYS USE BEFORE CLAIMING FACTS
- **DeepWiki MCP** (`mcp__deepwiki__ask_question`): Verify against source code. Code is truth.
- **HuggingFace MCP** (`mcp__hf-mcp-server__*`): `hub_repo_details` for metadata, `hub_repo_search` for discovery, `paper_search` for literature.
- **WebSearch**: Cross-reference claims, verify citations exist.
- **Semantic Scholar API**: Citation graphs, paper metadata. Key: `4ttKZAJR5P29PSwmNwaWQzihDej2m5CHyym3k560`. `curl -H "x-api-key: $KEY" "https://api.semanticscholar.org/graph/v1/paper/..."`.

### Verification Protocol
1. Never claim a paper exists without verifying via WebSearch or S2 API
2. Never claim a dataset has specific fields without inspecting via HF MCP
3. Never claim a model has specific capabilities without checking the model card
4. Fabricated or unverifiable → say so explicitly, no hedging

## Key Files
- `pilot/LOG.md` — Experimental log (runs 1-13 + current objective & scope)
- `docs/distill/reasoning-strategy-diversity.md` — Full distillation (7 principles, 10 insights, sources)
- `docs/distill/notes/principles.md` — Deutsch vary-test on all 8 candidate principles
- `docs/distill/notes/sources.md` — 28 sources with Lindy annotations
- `docs/HANDBOOK.md` — Discovery index
- `memory/MEMORY.md` — Cross-session persistent memory

## Infrastructure
- Modal CLI: `/Users/duy/Library/Python/3.9/bin/modal`
- GPU: A10G on Modal
- Budget: ~$74 remaining
- Key script: `pilot/modal_strategy_diversity.py`
