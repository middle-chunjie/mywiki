---
type: source
subtype: paper
title: "To Infinity and Beyond: Tool-Use Unlocks Length Generalization in State Space Models"
slug: malach-2026-infinity-2510-14826
date: 2026-04-20
language: en
tags: [state-space-model, tool-use, length-generalization, reasoning, coding]
processed: true

raw_file: raw/papers/malach-2026-infinity-2510-14826/paper.pdf
raw_md: raw/papers/malach-2026-infinity-2510-14826/paper.md
bibtex_file: raw/papers/malach-2026-infinity-2510-14826/paper.bib
possibly_outdated: false

authors:
  - Eran Malach
  - Omid Saremi
  - Sinead Williamson
  - Arwen Bradley
  - Aryo Lotfi
  - Emmanuel Abbe
  - Josh Susskind
  - Etai Littwin
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2510.14826
doi:
url: https://arxiv.org/abs/2510.14826
citation_key: malach2026infinity
paper_type: theory

read_status: unread

domain: agents
---

## Summary

The paper studies when state space models (SSMs) can generalize to much longer outputs than those seen in training. It first formalizes long-form generation via growing output support and proves a negative result: any generalized SSM with fixed internal memory fails on sufficiently complex long-form tasks in pure chain-of-thought or single-turn tool-use settings. It then gives a constructive positive result showing that interactive external-memory tools can restore length generalization for any tractable task. Experiments support the theory: tool-augmented Mamba, LSTM, and GRU models extrapolate strongly on arithmetic, logical reasoning, and a synthetic coding task, while Transformer baselines degrade more sharply outside the training distribution.

## Problem & Motivation

Transformers remain strong for reasoning and sequence generation, but their attention cost grows quadratically with sequence length, which makes long-context and long-form generation expensive. SSMs offer linear-time sequence processing and bounded internal memory, so they are attractive for long outputs, agent loops, and tool-rich settings. The paper asks whether that efficiency comes with an unavoidable expressivity gap on tasks whose output complexity grows with input complexity, and whether interactive tool use can compensate for the memory bottleneck enough to recover robust length generalization.

## Method

- **Task formalization**: defines a long-form generation task using the effective output support `supp_α(f(D_n))`, requiring it to grow monotonically with complexity `n` and satisfy `lim_{n->∞} supp_α(f(D_n)) = ∞`.
- **Model class**: defines a generalized state space model (GSSM) with finite state space `|S| < ∞`, update rule `u: S x Σ -> S`, and output rule `r: S -> Δ(Σ)`, covering recurrent models and fixed-window attention systems.
- **Negative theorem**: proves that for any CoT-only or single-turn GSSM there exists `n_0` such that `err_n(h) >= 1 - α` for all `n >= n_0`, because bounded internal state cannot represent the growing support of valid outputs.
- **Positive theorem**: introduces an interactive external-memory oracle and shows that, with problem-dependent trajectories, a simple GSSM learning algorithm can length-generalize on any computationally tractable long-form task by simulating Turing-machine-style read/write computation.
- **Tool interfaces**: uses pointer-based memory commands such as `read`, `move_left`, and `move_right` for arithmetic and Hanoi, plus a search-based memory command `find(x)` for logical-graph reasoning; thoughts and outputs are appended to memory, while commands and observations are not.
- **Training objective**: trains models with next-token prediction and teacher forcing, masking the input question and observation tokens from the loss; synthetic trajectories explicitly encode the target algorithm.
- **Synthetic model configs**: compares Mamba-130M (`24` layers, `d_model = 768`, intermediate size `1536`), LSTM (`4` layers, hidden size `1536`), GRU (`4` layers, hidden size `1536`), Pythia-style Transformer (`24` layers, `8` heads, `d_model = 768`, intermediate size `1536`), and a scaled-down Mistral-style model (`8` layers, `d_model = 768`, sliding-window `512`).
- **Optimization details**: synthetic runs search `learning_rate in {1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3}`, `batch_size in {128, 256, 512, 1024}`, `weight_decay in {0, 0.01}`, with `2,000` training steps; the coding experiment finetunes pretrained Mamba-1.4B and Pythia-1.4B at `learning_rate = 1e-4`, `weight_decay = 0.01`, `batch_size = 512`, `200` steps, context length `8,192`, using about `100K` successful trajectories per agent type on `8 x H100`.

## Key Results

- **Theory**: Theorem 2.1 shows any CoT-only or single-turn GSSM eventually incurs `err_n(h) >= 1 - α`, while Theorem 2.2 shows interactive tool-use is sufficient for length generalization on any tractable long-form task under the paper's training construction.
- **Addition**: Mamba and LSTM trained on demonstrations up to `5` digits generalize perfectly to `1,000`-digit addition; the paper reports that a Transformer trained in the same setting fails to extrapolate comparably.
- **Synthetic benchmarks**: Mamba reaches `10->1K (100%)` on `n x 1` multiplication, `10->1K (100%)` on `n x 2` multiplication, `10->1K (98%)` on Logical Graph, and `8->12 (49%)` on Tower of Hanoi.
- **Baseline comparison**: Pythia is much weaker out of distribution, dropping to `10->20 (79%)` on `n x 1`, `10->14 (12%)` on `n x 2`, and only `10->1K (5%)` on Logical Graph; Mistral also trails Mamba on arithmetic extrapolation.
- **Coding extrapolation**: in the synthetic code-fixing environment, models are trained on codebases up to `16` files; Transformer distillation exceeds `90%` pass rate on small codebases, but Mamba retains stronger performance on larger codebases when trained on interactive trajectories rather than single-turn demonstrations.

## Limitations

- The positive result depends on an interactive external-memory oracle and carefully constructed task-specific trajectories; it does not show that arbitrary off-the-shelf SSM training will discover such behavior on its own.
- Most empirical evidence comes from synthetic arithmetic, logical-graph, Hanoi, and procedurally generated coding environments rather than naturalistic open-ended agent benchmarks.
- The coding task is still narrow: the bug is structurally simple, successful trajectories are filtered, and the environment is far easier than real software engineering repositories.
- The theoretical positive result analyzes a simple string-matching-style learner rather than standard gradient-based optimization, so the gap between constructive possibility and practical learnability remains partially unresolved.
- Tower of Hanoi extrapolation is notably weaker, with output length growing as `2^n - 1`, showing that interactive tool use does not eliminate all optimization and robustness issues.

## Concepts Extracted

- [[state-space-model]]
- [[long-form-generation]]
- [[length-extrapolation]]
- [[tool-augmented-language-model]]
- [[react]]
- [[chain-of-thought]]
- [[teacher-forcing]]
- [[reasoning-language-model]]
- [[code-repair]]
- [[dependency-graph]]

## Entities Extracted

- [[eran-malach]]
- [[omid-saremi]]
- [[sinead-williamson]]
- [[arwen-bradley]]
- [[aryo-lotfi]]
- [[emmanuel-abbe]]
- [[josh-susskind]]
- [[etai-littwin-apple]]
- [[apple]]
- [[mamba]]
- [[swe-agent]]
- [[the-pile]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
