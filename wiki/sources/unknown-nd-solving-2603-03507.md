---
type: source
subtype: paper
title: Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-Based Self-Verification
slug: unknown-nd-solving-2603-03507
date: 2026-04-20
language: en
tags: [llm, math-reasoning, self-verification, code-interpreter, prompting]
processed: true

raw_file: raw/papers/unknown-nd-solving-2603-03507/paper.pdf
raw_md: raw/papers/unknown-nd-solving-2603-03507/paper.md
bibtex_file: raw/papers/unknown-nd-solving-2603-03507/paper.bib
possibly_outdated: false

authors:
  - Aojun Zhou
  - Ke Wang
  - Zimu Lu
  - Weikang Shi
  - Sichun Luo
  - Zipeng Qin
  - Shaoqing Lu
  - Anya Jia
  - Linqi Song
  - Mingjie Zhan
  - Hongsheng Li
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2603.03507
doi:
url: https://arxiv.org/abs/2603.03507
citation_key: unknownndsolving
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper studies why GPT-4 Code Interpreter is markedly stronger than text-only prompting on hard math word problems, then converts that analysis into a prompting method. It identifies frequent code generation, execution, and post-execution repair as the main drivers of performance, and proposes explicit code-based self-verification (CSV), which asks the model to solve, verify, and, when verification fails, revise the solution with additional code. The paper further adds verification-guided weighted majority voting over multiple sampled solutions. On MATH, GPT4-Code improves from `69.69%` to `73.54%` with CSV alone and to `84.32%` with CSV plus weighted voting, while also improving GSM8K and MMLU-Math in zero-shot settings.

## Problem & Motivation

The paper targets the gap between strong general-purpose LLM reasoning and reliable mathematical problem solving. Text-only chain-of-thought often fails on arithmetic precision, symbolic manipulation, and answer checking, while prior code-assisted methods such as PAL usually call code only once and lack a built-in mechanism to recover from execution errors or logically incorrect final answers. The authors therefore ask two questions: what aspect of GPT-4 Code Interpreter actually drives its math gains, and can that same mechanism be exploited to make the model explicitly verify and repair its own solutions without relying on an external verifier model or handcrafted few-shot examples?

## Method

- **Prompt-based analysis setup**: the paper compares three prompting regimes on GPT4-Code: Prompt 1 forbids code, Prompt 2 permits exactly one code block, and the Basic Prompt leaves code unrestricted. It defines **Code Usage Frequency** as the number of code executions in a solution trace and uses it as a measurable proxy for how much the model relies on executable computation.
- **Mechanistic finding**: unrestricted GPT4-Code interleaves natural-language reasoning and short code snippets, then uses execution feedback to revise buggy or implausible steps. The paper characterizes this adaptive correction behavior as [[self-debugging]] and argues that it is a core reason Basic Prompt outperforms one-shot code prompting.
- **Explicit code-based self-verification (CSV)**: after producing a candidate solution `C`, the model is prompted to generate verification code and assign a verification state `V ∈ {True, False, Uncertain}`. The process is summarized as `C -> V`, with `False -> C_new -> V -> ... -> final answer`, so a failed verification triggers solution revision instead of immediate termination.
- **Zero-shot implementation**: CSV is intentionally lightweight. It uses a direct instruction such as solving the problem step by step with Code Interpreter and verifying the final answer using code, rather than training an external verifier or crafting task-specific few-shot demonstrations.
- **Verification-guided weighted majority voting**: from `k` sampled solutions, the framework collects answer-state pairs `(v^i, a^i)` and scores each answer by `Score(a) = Σ_v w_v * #{i | a^i = a and v^i = v}`, where `v ∈ {True, Uncertain, False}` and `w_T > w_U > w_F`. The final prediction is `argmax_a Score(a)`.
- **Sampling configuration**: on MATH, the strongest setting uses CSV plus weighted voting with `k = 16` sampled paths. The paper also studies smaller sampled-path settings on GSM8K and analyzes how the weighting scheme behaves under different `w_T`, `w_U`, and `w_F` choices.
- **Evaluation scope**: experiments cover MATH, GSM8K, MMLU-Math, and MMLU-STEM, plus ablations comparing code-based verification against natural-language verification and analyses of verification precision/recall, difficulty level, subject category, and Python package usage.

## Key Results

- On **MATH**, GPT-4 scores `42.20%`, GPT-4 (CoT) `50.36%`, GPT-4 (PHP) `53.90%`, and GPT4-Code `69.69%`; adding CSV raises GPT4-Code to `73.54%`, and adding CSV plus verification-guided weighted majority voting reaches `84.32%` with `k = 16`.
- The largest MATH subtopic gains after full voting are substantial: Intermediate Algebra `50.1 -> 74.4`, Number Theory `77.2 -> 94.1`, Counting & Probability `70.6 -> 89.0`, and Algebra `83.6 -> 95.6`. Geometry improves less under CSV alone (`53.4 -> 54.0`) and reaches `64.9` after voting.
- On **GSM8K**, GPT4-Code improves from `92.9%` to `97.0%` with CSV plus voting using only `5` sampled paths. On **MMLU-Math**, accuracy rises from `87.5%` to `89.2%`; on **MMLU-STEM**, from `86.8%` to `87.0%`.
- Code-based verification is materially better than natural-language verification: the natural-language verifier variant drops overall MATH accuracy to `69.29%`, below the `69.69%` Basic Prompt baseline, whereas code-based CSV improves all seven MATH subtopics and lifts overall accuracy by `+3.85` points.
- Verification quality is strong enough to support weighted voting: the paper reports average verification precision of `95.88%`, plus gains of `22.3%` in average precision and `5.6%` in average recall relative to raw accuracy-oriented counting.
- In the appendix analysis of code traces on MATH, `sympy` is the most frequently used package with usage frequency `0.4168`, followed by `math` at `0.1590`, indicating that symbolic and numerical computation are central to the successful reasoning trajectories.

## Limitations

- The analysis and prompting improvements are centered on GPT4-Code, so transfer to other LLMs or other tool-use environments is not established.
- Geometry remains a weak area: CSV alone yields only a `+0.6` point gain on the Geometry subset, and the paper attributes part of this gap to the lack of multimodal reasoning for diagram-heavy problems.
- Verification-guided voting increases inference cost because it requires repeated solution sampling and multiple code-execution rounds rather than a single-pass answer.
- The method improves benchmark accuracy, but the paper provides limited deeper analysis of failure cases for the `Uncertain` state, notebook-state robustness, or broader tool-execution safety concerns.

## Concepts Extracted

- [[large-language-model]]
- [[mathematical-reasoning]]
- [[code-generation]]
- [[code-execution]]
- [[code-interpreter]]
- [[self-debugging]]
- [[self-verification]]
- [[chain-of-thought-prompting]]
- [[zero-shot-prompting]]
- [[self-consistency]]
- [[code-usage-frequency]]
- [[explicit-code-based-self-verification]]
- [[verification-guided-weighted-majority-voting]]

## Entities Extracted

- [[aojun-zhou]]
- [[ke-wang]]
- [[zimu-lu]]
- [[weikang-shi]]
- [[sichun-luo]]
- [[zipeng-qin]]
- [[shaoqing-lu]]
- [[anya-jia]]
- [[linqi-song]]
- [[mingjie-zhan]]
- [[hongsheng-li]]
- [[gpt-4]]
- [[openai]]
- [[math-dataset]]
- [[gsm8k]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
