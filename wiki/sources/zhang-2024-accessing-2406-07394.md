---
type: source
subtype: paper
title: Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B
slug: zhang-2024-accessing-2406-07394
date: 2026-04-20
language: en
tags: [llm, mathematical-reasoning, mcts, self-refinement, planning]
processed: true
raw_file: raw/papers/zhang-2024-accessing-2406-07394/paper.pdf
raw_md: raw/papers/zhang-2024-accessing-2406-07394/paper.md
bibtex_file: raw/papers/zhang-2024-accessing-2406-07394/paper.bib
possibly_outdated: false
authors:
  - Di Zhang
  - Xiaoshui Huang
  - Dongzhan Zhou
  - Yuqiang Li
  - Wanli Ouyang
year: 2024
venue: arXiv preprint
venue_type: preprint
arxiv_id: 2406.07394
doi:
url: http://arxiv.org/abs/2406.07394
citation_key: zhang2024accessing
paper_type: method
read_status: unread
domain: llm
---

## Summary

This paper introduces MCT Self-Refine (MCTSr), a framework that integrates Monte Carlo Tree Search with LLM self-refinement and self-evaluation to enhance mathematical reasoning. Nodes in the search tree represent answer versions; edges represent self-refinement attempts. Each node is scored by an LLM self-reward mechanism under strict constraints, and quality is propagated via a custom Q-value formula blending minimum and mean rewards. An improved UCT formula governs node selection. Using LLaMA-3 8B as the backbone, MCTSr achieves GPT-4-comparable accuracy on several benchmarks: 96.66% on GSM8K, 58.24% on MATH, and 49.36% on GAIC Math Odyssey with 8 rollouts, substantially exceeding the same model's zero-shot CoT baseline of 74.07%, 24.36%, and 17.22% respectively.

## Problem & Motivation

LLMs suffer from hallucination and unreliable step-by-step reasoning, especially for multi-step Olympiad-level mathematics. Naive self-refinement (one pass) helps only marginally. Prior MCTS integrations with LLMs lacked a principled way to define "states" and "actions" in a continuous generative space, and did not leverage the model's self-reward capability for tree-node value estimation.

## Method

- **Tree structure**: Nodes are answer strings; edges are self-refinement operations (feedback-guided rewriting). Root initialized with either a naive model answer or a dummy "I don't know" response to prevent overfitting.
- **Self-Refine step**: Given answer `a` to problem `P`, model first produces critical feedback `m`, then rewrites `a` guided by `m` to produce `a'`. Prompt enforces step-by-step reasoning and a `[Final Answer]` marker.
- **Self-Evaluation**: Model scores its own answer in `[-100, 100]`. Three constraints applied:
  - Strict scoring prompt to prevent grade inflation.
  - Full-score suppression: any score `>95` reduced by a constant.
  - Repeated sampling per node visit; parent node also re-sampled when children are sampled.
- **Q-value formula**: `Q(a) = 0.5 * (min(R_a) + mean(R_a))` — blends worst-case and average rewards to sharpen discrimination between answers.
- **Backpropagation**: If any child's Q changes, parent Q is updated as `Q'(a) = 0.5 * (Q(a) + max_{i in Children(a)} Q(i))`.
- **UCT formula**: `UCT_a = Q(a) + c * sqrt( ln(N(Father(a)) + 1) / (N(a) + eps) )` — adapted from UCB-1; `eps` avoids division by zero.
- **"Full expansion" criterion**: A node is fully expanded when (1) child count reaches a predefined limit AND (2) at least one child's Q exceeds the node's Q. Only non-fully-expanded nodes and leaf nodes form the candidate set `C`.
- **Termination**: Early stopping on diminishing improvement, rollout count limit, or maximum depth.
- **Backbone**: LLaMA-3 8B with zero additional fine-tuning.

## Key Results

- **GSM8K**: Zero-shot CoT 74.07% → 8-rollout MCTSr 96.66% (vs GPT-4 Turbo 97.1%, Claude 3 Opus 95%).
- **MATH (all levels)**: Zero-shot CoT 24.36% → 8-rollout MCTSr 58.24% (vs GPT-4 Turbo 73.4%, Gemini 1.5-Pro 67.7%).
  - Level 1: 90.16%; Level 5: 34.06% (hardest).
- **GSM-Hard**: Zero-shot 25.47% → 8-rollout MCTSr 45.49%.
- **AIME (1983–2024)**: 2.36% → 11.79% with 8 rollouts (933 problems).
- **GAIC Math Odyssey**: 17.22% → 49.36% with 8 rollouts (389 problems, minimal pre-training overlap).
- **OlympiadBench**: 1.25% → 7.76% with 8 rollouts.
- Clear monotonic improvement with rollout count across all datasets.

## Limitations

- MCTSr is compute-intensive: each rollout requires multiple LLM calls (self-refine + repeated self-reward sampling), making wall-clock cost orders of magnitude higher than zero-shot CoT.
- Self-evaluation reliability depends on the model's internal calibration; weaker models may produce noisy reward signals that degrade search quality.
- Still well below GPT-4 on MATH (58% vs 73%) and Olympiad tasks; the gap grows at higher difficulty levels.
- Broader applicability (black-box optimization, alignment) is stated as future work but not demonstrated.
- No ablation on the impact of the dummy root initialization or the specific Q-value blending formula.

## Concepts Extracted

- [[monte-carlo-tree-search]]
- [[self-refinement]]
- [[mathematical-reasoning]]
- [[large-language-model]]
- [[chain-of-thought-prompting]]
- [[exploration-exploitation-tradeoff]]
- [[upper-confidence-bound]]
- [[self-evaluation]]
- [[process-reward-model]]
- [[tree-of-thought]]

## Entities Extracted

- [[di-zhang-shai]]
- [[xiaoshui-huang]]
- [[dongzhan-zhou]]
- [[yuqiang-li]]
- [[wanli-ouyang]]
- [[llama-3-8b]]
- [[gpt-4]]
- [[gemini-1-5-pro]]
- [[gsm8k]]
- [[math-dataset]]
- [[aime]]
- [[olympiadbench]]
- [[shanghai-artificial-intelligence-laboratory]]
- [[fudan-university]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
