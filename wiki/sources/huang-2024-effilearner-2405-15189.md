---
type: source
subtype: paper
title: "EffiLearner: Enhancing Efficiency of Generated Code via Self-Optimization"
slug: huang-2024-effilearner-2405-15189
date: 2026-04-20
language: en
tags: [llm, code-generation, efficiency, profiling, self-optimization]
processed: true

raw_file: raw/papers/huang-2024-effilearner-2405-15189/paper.pdf
raw_md: raw/papers/huang-2024-effilearner-2405-15189/paper.md
bibtex_file: raw/papers/huang-2024-effilearner-2405-15189/paper.bib
possibly_outdated: false

authors:
  - Dong Huang
  - Jianbo Dai
  - Han Weng
  - Puzhen Wu
  - Yuhao Qing
  - Heming Cui
  - Zhijiang Guo
  - Jie M. Zhang
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2405.15189
doi:
url: http://arxiv.org/abs/2405.15189
citation_key: huang2024effilearner
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

This paper studies how to make large language model code generation more efficient after the model has already produced functionally correct code. EFFI-LEARNER runs the generated program on open test cases, collects line-level execution-time and memory-use profiles, and feeds those overhead traces back into the LLM for iterative self-optimization. The framework is evaluated on EffiBench, HumanEval, and MBPP across 16 open-source and 6 closed-source models. The main finding is that profiler-guided refinement consistently reduces runtime and memory cost far more than naive self-refinement or scalar-only feedback. Representative gains include StarCoder2-15B on EffiBench dropping from `0.93 s` to `0.12 s` execution time and from `22.02 Mb*s` to `2.03 Mb*s` total memory usage, while incurring only minor pass@1 degradation.

## Problem & Motivation

Existing code-generation work mostly optimizes for functional correctness, but practically useful generated programs also need low runtime and memory overhead. Prior evidence from EffiBench shows that even strong LLMs can produce code that passes tests while remaining substantially slower or more memory-hungry than canonical solutions. The paper's core motivation is that human programmers often optimize by reading execution profiles, locating hot lines or wasteful allocations, and then rewriting code accordingly. EFFI-LEARNER transfers that workflow to LLMs: instead of asking a model to "optimize" in the abstract, it supplies concrete overhead evidence tied to specific code lines.

## Method

- **Overall loop**: generate an initial program from the task description, execute it locally on open test cases, profile overhead, and feed the task description, test case, original code, and profiler output back to the LLM for refinement.
- **Execution-time profiling**: uses Python `line_profiler` to collect per-line runtime statistics over all open test cases combined, including line numbers, execution counts, and total time spent at each line.
- **Memory profiling**: uses Python `memory_profiler` to record line-level memory behavior over the same open test cases, exposing where loops, function calls, or allocations create unnecessary memory cost.
- **Refinement prompt**: the self-optimization prompt contains `task_description`, `test_case`, `completion`, and `overhead_prompt`, explicitly asking the model to improve efficiency while still passing the provided test.
- **Optimization targets**: the model is encouraged to apply transformations such as algorithm substitution, loop restructuring, data-structure optimization, memoization, and code simplification, while preserving correctness.
- **Iteration budget**: the framework is iterative rather than single-shot; ablations evaluate `0` to `5` self-optimization steps and show the first step gives the largest gains, with smaller improvements afterward.
- **Evaluation metrics**: efficiency is measured with `ET = (1/N) Σ T_code`, `NET = (1/N) Σ (T_code / T_canonical)`, `MU = (1/N) Σ M_code`, `NMU = (1/N) Σ (M_code / M_canonical)`, `TMU = (1/N) Σ ∫_0^T M(t) dt`, and `NTMU = (1/N) Σ (TMU_code / TMU_canonical)`.
- **Evaluation protocol**: optimization uses open tests, while final reporting uses private tests for EffiBench and EvalPlus-style private suites for HumanEval and MBPP; only initially correct solutions are included so the method targets efficiency, not pass@1 improvement.
- **Compute setup**: experiments run on a server with an Intel Xeon Platinum 8336C CPU (`128` cores), `8 × NVIDIA A100-SXM` GPUs, and `2.0 TiB` total memory.

## Key Results

- On EffiBench, **StarCoder2-15B** improves from `ET = 0.93 s` to `0.12 s` (`-87.1%`), `NET = 7.58` to `1.03` (`-86.4%`), and `TMU = 22.02 Mb*s` to `2.03 Mb*s` (`-90.8%`).
- On EffiBench, **DeepSeek-6.7B-Ins** reduces `MU` from `259.73 Mb` to `36.97 Mb` (`-85.8%`) and `TMU` from `555.18 Mb*s` to `13.66 Mb*s` (`-97.5%`).
- For a closed model, **GPT-3.5-Turbo-0301** improves from `ET = 0.36 s` to `0.28 s` (`-22.2%`), `MU = 91.25 Mb` to `36.08 Mb` (`-60.5%`), and `TMU = 157.50 Mb*s` to `12.43 Mb*s` (`-92.1%`).
- The number-of-steps ablation shows diminishing returns: for **CodeLlama-70B**, the first optimization step already cuts `MU` from `109.61 Mb` to `26.47 Mb` (`-75.9%`), while five steps reach `ET = 0.47 s` and `TMU = 14.53 Mb*s`.
- Rich profiler feedback outperforms simpler refinement signals: for **CodeLlama-70B**, unsupervised self-refine worsens `TMU` from `203.92` to `1261.83 Mb*s`, whereas EFFI-LEARNER reduces it to `14.53 Mb*s`.
- Generalization holds beyond EffiBench: on HumanEval, **CodeLlama-70B** drops from `0.21 s` to `0.18 s` execution time (`-14.3%`), and similar improvements are reported on MBPP.
- Correctness can drop slightly, but modestly: the paper reports pass@1 decreases of roughly `0.0` to `0.5` points because optimization is guided only by open tests.

## Limitations

- The multi-iteration optimization loop adds latency and token cost because full overhead profiles must be generated and inserted into prompts.
- Evaluation is primarily on Python, so transfer to other languages, runtimes, or profiling ecosystems is not established.
- The method assumes the initial solution is already correct on open tests; it is not designed to recover badly wrong code.
- Because optimization uses public/open tests while final correctness is judged on private tests, pass@1 can decline slightly after refinement.
- Gains are not uniform: some already near-optimal solutions show minimal improvement, and some metrics can worsen in edge cases, such as StarCoder2-15B's `MU` increasing from `26.35 Mb` to `27.67 Mb`.

## Concepts Extracted

- [[large-language-model]]
- [[code-generation]]
- [[self-optimization]]
- [[overhead-profiling]]
- [[execution-time-profiling]]
- [[memory-profiling]]
- [[iterative-refinement]]
- [[functional-correctness]]
- [[prompt-engineering]]

## Entities Extracted

- [[dong-huang]]
- [[jianbo-dai]]
- [[han-weng]]
- [[puzhen-wu]]
- [[yuhao-qing]]
- [[heming-cui]]
- [[zhijiang-guo]]
- [[jie-m-zhang]]
- [[effibench]]
- [[humaneval]]
- [[mbpp]]
- [[line-profiler]]
- [[memory-profiler]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
