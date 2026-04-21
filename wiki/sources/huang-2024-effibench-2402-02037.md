---
type: source
subtype: paper
title: "EffiBench: Benchmarking the Efficiency of Automatically Generated Code"
slug: huang-2024-effibench-2402-02037
date: 2026-04-20
language: en
tags: [code-generation, efficiency, benchmark, llm, software-engineering]
processed: true

raw_file: raw/papers/huang-2024-effibench-2402-02037/paper.pdf
raw_md: raw/papers/huang-2024-effibench-2402-02037/paper.md
bibtex_file: raw/papers/huang-2024-effibench-2402-02037/paper.bib
possibly_outdated: false

authors:
  - Dong Huang
  - Yuhao Qing
  - Weiyi Shang
  - Heming Cui
  - Jie M. Zhang
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.02037
doi:
url: https://arxiv.org/abs/2402.02037
citation_key: huang2024effibench
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

EffiBench is a benchmark for measuring the efficiency, not just correctness, of LLM-generated code. The benchmark contains 1,000 efficiency-critical problems drawn from LeetCode, each paired with an executable human-written canonical solution and a test-case generator designed to stress runtime and memory behavior under diverse workloads. The paper evaluates 21 code generation models, including 13 open-source and 8 closed-source LLMs, using execution-time and memory-based metrics such as ET, NET, MU, NMU, TMU, and NTMU. GPT-4-turbo is the strongest model overall, but it still trails the canonical solutions: its average execution time is `1.69x` and its total memory usage is `3.18x` the baseline, with much worse worst-case outliers.

## Problem & Motivation

Prior code-generation benchmarks mostly emphasize functional correctness, but two correct programs can have very different algorithmic and systems-level efficiency. The paper argues that existing datasets such as HumanEval, MBPP, and APPS are poorly suited for this question because many tasks are too simple, not explicitly efficiency-critical, and do not provide enough heavy or diverse test cases to expose runtime and memory gaps. EffiBench is motivated by the need for a benchmark that makes these efficiency differences visible and measurable at scale.

## Method

- **Problem collection pipeline**: start from all HuggingFace problems tagged with LeetCode, deduplicate by LeetCode problem ID, and filter out problems with interview frequency below `40%`. This produces `2,605` initial candidates.
- **Efficiency-critical filtering**: map each problem back to LeetCode topic tags, then retain problems from `11` common algorithm families such as greedy, dynamic programming, DFS, BFS, binary search, and sorting. This narrows the pool to `1,146` candidates.
- **Canonical solution construction**: collect top-starred solutions from the LeetCode discussion forum, manually repair environment-specific dependencies such as `TreeNode`, `ListNode`, `List`, and `Bisect`, and discard solutions that depend on LeetCode-only packages. The final benchmark contains `1,000` executable canonical solutions.
- **Test generation**: use `GPT-3.5-turbo` to produce per-problem test-case generators covering varied input sizes, data distributions, and edge cases. EffiBench ships with `100` tests per problem by default, and the appendix also studies `10` and `1,000` tests.
- **Efficiency metrics**: define execution-time and memory metrics as `ET = (1/N) Σ T_code`, `NET = (1/N) Σ (T_code / T_canonical)`, `MU = (1/N) Σ M_code`, `NMU = (1/N) Σ (M_code / M_canonical)`, `TMU = (1/N) Σ ∫_0^{T_total} M(t) dt`, and `NTMU = (1/N) Σ (TMU_code / TMU_canonical)`.
- **Evaluation protocol**: benchmark `13` open-source and `8` closed-source LLMs, using an MBPP-style prompt with task description, input/output examples, explanations, and assertion constraints. Results are collected on Ubuntu 20.04 with an Intel Xeon Silver 4216 CPU, `32GB` RAM, Python `3.8.11`, and averaged over `5` executions.

## Key Results

- EffiBench covers `1,000` problems across `11` algorithm subsets, with `171` easy, `589` medium, and `240` hard tasks.
- GPT-4-turbo is the best overall model in the main comparison table: `ET = 0.20 s`, `NET = 1.69`, `MU = 37.06 Mb`, `NMU = 1.24`, `TMU = 14.36 Mb*s`, `NTMU = 3.18`, and `Pass@1 = 60.2`.
- Even for GPT-4-turbo, worst-case efficiency outliers remain severe: `max NET = 45.49` and `max NTMU = 142.89`.
- On the common subset of `113` problems solved correctly by all closed-source models, GPT-4-turbo remains strongest with `NET = 1.34` and `NTMU = 2.34`.
- Efficiency varies sharply by algorithm family. For `gpt-3.5-turbo-0301`, divide-and-conquer is relatively strong with `NET = 1.30` and `NTMU = 1.53`, while DFS and BFS are weak with `NTMU = 501.64` and `412.37`, respectively.
- Increasing the number of tests raises absolute efficiency penalties, but the relative ranking among models stays broadly consistent.

## Limitations

- The benchmark is built from LeetCode-style Python tasks, so its coverage of other programming languages and real-world software engineering settings is limited.
- Efficiency is measured only on correctly generated code, which couples the analysis to model correctness and reduces comparability when different models solve different subsets.
- Absolute efficiency scores depend on test coverage; the appendix shows that moving from `10` to `1,000` tests can materially change NET/NMU/NTMU values.
- Canonical solutions are curated from top-starred forum answers and manually repaired, which is practical but still an imperfect proxy for globally optimal implementations.

## Concepts Extracted

- [[benchmark-dataset]]
- [[code-generation]]
- [[test-case-generation]]
- [[execution-based-evaluation]]
- [[functional-correctness]]
- [[code-efficiency]]
- [[canonical-solution]]
- [[normalized-execution-time]]
- [[total-memory-usage]]

## Entities Extracted

- [[dong-huang]]
- [[yuhao-qing]]
- [[weiyi-shang]]
- [[heming-cui]]
- [[jie-m-zhang]]
- [[effibench]]
- [[leetcode]]
- [[humaneval]]
- [[mbpp]]
- [[gpt-4-turbo]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
