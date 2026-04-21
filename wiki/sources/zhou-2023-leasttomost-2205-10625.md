---
type: source
subtype: paper
title: Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
slug: zhou-2023-leasttomost-2205-10625
date: 2026-04-20
language: en
tags: [prompting, reasoning, llm, few-shot, compositional-generalization]
processed: true
raw_file: raw/papers/zhou-2023-leasttomost-2205-10625/paper.pdf
raw_md: raw/papers/zhou-2023-leasttomost-2205-10625/paper.md
bibtex_file: raw/papers/zhou-2023-leasttomost-2205-10625/paper.bib
possibly_outdated: true
authors:
  - Denny Zhou
  - Nathanael Schärli
  - Le Hou
  - Jason Wei
  - Nathan Scales
  - Xuezhi Wang
  - Dale Schuurmans
  - Claire Cui
  - Olivier Bousquet
  - Quoc Le
  - Ed Chi
year: 2023
venue: ICLR 2023
venue_type: conference
arxiv_id: 2205.10625
doi:
url: http://arxiv.org/abs/2205.10625
citation_key: zhou2023leasttomost
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent LLM prompting literature.

Least-to-most (L2M) prompting addresses the easy-to-hard generalization failure of chain-of-thought (CoT) prompting by introducing a two-stage in-context strategy: first decompose the original problem into a list of simpler subproblems, then solve them sequentially so that each step's answer is conditioned on previous answers. Both stages use only few-shot demonstrations — no training or fine-tuning is needed. On SCAN compositional generalization with GPT-3 `code-davinci-002`, L2M achieves `99.7%` accuracy on the length split using just 14 exemplars, compared to `16.2%` for CoT. On GSM8K math word problems, L2M improves multi-step (≥5 steps) accuracy from `39.07%` to `45.23%`. The approach draws from educational-psychology least-to-most prompting and generalizes across symbolic manipulation, compositional generalization, and arithmetic reasoning.

## Problem & Motivation

Chain-of-thought prompting has shown impressive results on complex NLP tasks, but it fails to generalize from easy demonstration examples to harder test problems — specifically when test cases require more steps or longer sequences than those shown in the prompt. Standard prompting and CoT both collapse on the SCAN length split (CoT: `16.2%`) and on last-letter-concatenation with long lists. The gap mirrors a fundamental difference between human and machine reasoning: humans routinely solve problems harder than any they have studied, while ML models generalize poorly beyond the difficulty level of their training distribution.

## Method

Least-to-most prompting operates in two sequential stages, both implemented via few-shot prompting:

- **Stage 1 — Decomposition**: A prompt containing fixed exemplars demonstrates how to break a complex problem into an ordered list of simpler subproblems. For SCAN, 8 exemplars show command-decomposition; for math, the exemplar has the model write "Let's break down this problem: 1. … 2. …" before solving.
- **Stage 2 — Sequential subproblem solving**: The prompt consists of (1) fixed solution exemplars, (2) all previously answered subproblems with their solutions appended, and (3) the next subproblem to solve. Each newly generated answer is appended to the context before the next step, so later subproblems can reference prior answers.
- **Single-pass variant**: For GSM8K, decomposition and solving are merged into a single prompt (Table 9), reducing the two API calls to one.
- **Python intermediate representation**: For SCAN, prompts use Python notation (`"LOOK" * 2` instead of `"LOOK LOOK"`) to stay within the `2048`-token context limit; a post-processing script expands expressions. An auxiliary experiment shows `code-davinci-002` can also expand Python expressions directly at `99.7%` accuracy.
- **Base model**: `code-davinci-002` (GPT-3 family) is used throughout; `text-davinci-002` and `code-davinci-001` are ablated. `code-davinci-002` consistently outperforms text variants.
- **No training or fine-tuning at any stage**: all generalization is in-context.

## Key Results

- **Last-letter-concatenation** (symbolic manipulation, length generalization): L2M vs. CoT at list length 12 — `74.0%` vs. `31.8%`; at length 4 — `94.0%` vs. `84.2%`. Standard prompting: `0%` at all lengths.
- **SCAN compositional generalization (length split)**: L2M `99.7%` vs. CoT `16.2%` with `code-davinci-002`; L2M `76.0%` vs. CoT `0.0%` with `text-davinci-002`. Only 14 exemplars needed; neural-symbolic SOTA required the full 15,000-example training set.
- **SCAN full dataset (all splits)**: L2M accuracy remains `≥99%` across all splits.
- **GSM8K math word problems** (overall): L2M `62.39%` vs. CoT `60.87%`. For problems requiring `≥5` steps: `45.23%` vs. `39.07%`.
- **DROP reading comprehension** (numerical subset): L2M `82.45%` vs. CoT `74.77%` (non-football); `73.42%` vs. `59.56%` (football).

## Limitations

- Decomposition prompts are domain-specific and do not transfer across task types. A prompt designed for math word problems does not teach decomposition for commonsense reasoning (e.g., StrategyQA-style questions like "Did Aristotle use a laptop?").
- Even within the same domain (GSM8K), automatic decomposition is the bottleneck: nearly all hard problems can be solved if the correct decomposition is provided manually.
- The approach requires designing two sets of prompts (decomposition + solution) per domain, increasing prompt-engineering cost relative to standard CoT.
- The two-stage variant doubles API call count per problem compared to single-pass CoT.
- Performance still degrades on very long sequences (last-letter-concatenation with length 12 is `74%`, not `100%`); errors are predominantly concatenation mistakes (dropping or duplicating letters) rather than identification errors.

## Concepts Extracted

- [[least-to-most-prompting]]
- [[chain-of-thought-prompting]]
- [[few-shot-prompting]]
- [[task-decomposition]]
- [[compositional-generalization]]
- [[easy-to-hard-generalization]]
- [[symbolic-reasoning]]
- [[mathematical-reasoning]]
- [[self-consistency]]
- [[large-language-model]]
- [[in-context-learning]]
- [[sequence-to-sequence]]

## Entities Extracted

- [[denny-zhou]]
- [[nathanael-scharli]]
- [[le-hou]]
- [[jason-wei]]
- [[nathan-scales]]
- [[xuezhi-wang]]
- [[dale-schuurmans]]
- [[claire-cui]]
- [[olivier-bousquet]]
- [[quoc-le]]
- [[ed-chi]]
- [[google-deepmind]]
- [[scan-benchmark]]
- [[gsm8k]]
- [[drop-benchmark]]
- [[gpt-3]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
