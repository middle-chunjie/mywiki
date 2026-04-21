---
type: source
subtype: paper
title: "Lean-STaR: Learning to Interleave Thinking and Proving"
slug: unknown-nd-leanstar
date: 2026-04-20
language: en
tags: [theorem-proving, formal-mathematics, reasoning, lean, chain-of-thought]
processed: true

raw_file: raw/papers/unknown-nd-leanstar/paper.pdf
raw_md: raw/papers/unknown-nd-leanstar/paper.md
bibtex_file: raw/papers/unknown-nd-leanstar/paper.bib
possibly_outdated: false

authors:
  - Haohan Lin
  - Zhiqing Sun
  - Sean Welleck
  - Yiming Yang
year: 2025
venue: ICLR 2025
venue_type: conference
arxiv_id:
doi:
url: https://openreview.net/forum?id=SOWZ59UyNc
citation_key: unknownndleanstar
paper_type: method

read_status: unread

domain: llm
---

## Summary

Lean-STaR reframes Lean proof generation as interleaved informal reasoning and tactic prediction rather than tactic-only decoding. The paper first uses [[gpt-4]] to retrospectively annotate successful proof steps from [[mathlib]] with natural-language thoughts conditioned on the current [[proof-state]] and the ground-truth tactic, then fine-tunes [[internlm2-math-base-7b]] on `(state, thought, tactic)` triples to obtain Lean-CoT. It next applies [[expert-iteration]], keeping only sampled proof trajectories that [[lean]] verifies and retraining on the union of synthetic CoT data and successful self-generated proofs. On [[mini-f2f]], the base 7B system improves from `29.5%` to `34.8%` pass@32 and `36.1%` pass@64; with a stronger InternLM2-plus variant it reaches `45.4%` pass@32 and `46.3%` pass@64, establishing stronger results for thought-augmented [[formal-theorem-proving]].

## Problem & Motivation

Prior neural theorem provers in Lean mostly learn from formal proof code alone, mapping a current goal state directly to the next tactic. The authors argue that this misses useful informal reasoning that humans rely on while proving theorems, especially the intent behind each local proof step. Lean-STaR is motivated by the hypothesis that explicitly modeling those hidden intermediate thoughts can improve tactic selection, increase proof diversity, and make search more effective when combined with a symbolic verifier. The work also targets a practical gap: Lean lacks the aggressive automation available in systems like Isabelle, so step-wise assistance may be more realistic than whole-proof generation.

## Method

- **MDP formulation**: theorem proving is cast as an MDP `(\mathcal{S}, \mathcal{A}, P_a, R_a)` where states are proof states `s_i`, actions are tactics `a_i`, and Lean produces the transition to the next state.
- **Direct tactic baseline**: the SFT model learns `\pi_M(a | s)` from successful `(state, tactic)` pairs extracted from proof trajectories in LeanDojo Benchmark 4 v9, using `231,240` supervised examples.
- **Thought-augmented factorization**: Lean-STaR introduces a latent thought `t_i` before each tactic, with `\pi_M(a_i, t_i | s_i) = \pi_M(a_i | t_i, s_i)\pi_M(t_i | s_i)` and induced tactic distribution `\pi_M(a_i | s_i) = \sum_{t_i}\pi_M(a_i | t_i, s_i)\pi_M(t_i | s_i)`.
- **Retrospective rationale generation**: an oracle model, implemented with [[gpt-4]], is given `(s_i, a_i)` and asked to synthesize a non-hindsight rationale `t_i`; the authors sample `17,256` successful proof trajectories and annotate `52,438` thoughts after filtering out tactics containing newlines.
- **Lean-CoT training**: the base model [[internlm2-math-base-7b]] is fine-tuned for `1` epoch on thought-augmented triples, with learning rate warmed from `0` to `2 x 10^-5` over the first `20%` of steps and then decayed by a cosine schedule.
- **Expert iteration**: starting from Lean-CoT, the system samples proof trajectories `[(s_0, t_0, a_0), ... , (s_n, t_n, a_n)]`, keeps only successful proofs checked by [[lean]], deduplicates them, and retrains on `D_T \cup D_1`; the reward is `R(...) = 1` iff a proof trajectory succeeds, and the expected objective is `J(M, D) = \sum_i \mathbb{E}_{\pi_M}[R(...)]`.
- **Iteration hyperparameters**: in each expert-iteration round the model samples `K = 32` trajectories per problem with temperature `T = 1.0`, limits tactic generation to `N = 5` per problem, caps each problem at `1` minute, averages about `0.5` A100 minutes per problem, and collects `32,231` state-thought-tactic pairs in the first round and roughly `19k` in the second.
- **Inference and evaluation**: because best-first search performs poorly when thoughts are present, the paper evaluates the thought-augmented models with [[sampling-based-proof-search]] using `N = 50`, `K = 32` or `64`, and `T = 0.7`, while comparing against search baselines with `S = 32`, `K = 1`.

## Key Results

- On [[mini-f2f]] with InternLM2-Math-base-7b, pass@32 improves from `28.7%` few-shot to `29.5%` SFT, `32.8%` Lean-CoT, `34.0%` Lean-STaR Iter-1, and `34.8%` Lean-STaR Iter-2.
- On the same base model, increasing sampling from pass@32 to pass@64 raises Lean-STaR Iter-2 from `34.8%` to `36.1%`.
- On the LeanDojo test subset, the base model improves from `29.7%` to `30.6%` after SFT, `35.6%` after Lean-CoT, `38.4%` after the first expert-iteration round, and `39.4%` after the second.
- With InternLM2-plus-7b, pass@32 on [[mini-f2f]] rises from `40.9%` few-shot to `41.3%` SFT, `43.4%` Lean-CoT, and `45.4%` Lean-STaR; pass@64 reaches `46.3%`.
- The gains are not explained by expert iteration alone: on the plus model, expert iteration without CoT reaches `43.0%`, below `45.5%` for Lean-STaR under the same `N = 50`, `K = 32`, `T = 0.7` setting.

## Limitations

- The method depends on a strong external oracle model (`GPT-4` / `GPT-4o`) to synthesize high-quality training thoughts, so data quality and reproducibility are partly tied to closed models.
- The experiments are centered on [[lean]], [[mathlib]], [[mini-f2f]], and [[lean-dojo]]; the paper does not show transfer to other theorem provers such as Isabelle or Coq.
- Best-first search with log-probability scoring works poorly once natural-language thoughts are introduced, so the framework currently relies on a sampling-based evaluator rather than a unified search procedure.
- Expert iteration is computationally expensive: the first round alone takes about `4` days on `8 x A100` GPUs and averages about `0.5` A100 minutes per problem.
- Generated thoughts can contain factual or arithmetic mistakes; correctness is recovered only because the theorem prover checks tactics and symbolic tools discharge some subproblems.

## Concepts Extracted

- [[formal-theorem-proving]]
- [[chain-of-thought]]
- [[retrospective-rationale-generation]]
- [[tactic-prediction]]
- [[proof-state]]
- [[expert-iteration]]
- [[interactive-theorem-proving]]
- [[sampling-based-proof-search]]
- [[reinforcement-learning]]
- [[formal-verification]]

## Entities Extracted

- [[haohan-lin]]
- [[zhiqing-sun]]
- [[sean-welleck]]
- [[yiming-yang]]
- [[carnegie-mellon-university]]
- [[tsinghua-university]]
- [[lean]]
- [[mathlib]]
- [[mini-f2f]]
- [[lean-dojo]]
- [[internlm2-math-base-7b]]
- [[gpt-4]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
