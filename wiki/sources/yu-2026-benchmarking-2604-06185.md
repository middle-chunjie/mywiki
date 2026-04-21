---
type: source
subtype: paper
title: Benchmarking LLM Tool-Use in the Wild
slug: yu-2026-benchmarking-2604-06185
date: 2026-04-20
language: en
tags: [agents, llm, benchmark, tool-use, evaluation]
processed: true

raw_file: raw/papers/yu-2026-benchmarking-2604-06185/paper.pdf
raw_md: raw/papers/yu-2026-benchmarking-2604-06185/paper.md
bibtex_file: raw/papers/yu-2026-benchmarking-2604-06185/paper.bib
possibly_outdated: false

authors:
  - Peijie Yu
  - Wei Liu
  - Yifan Yang
  - Jinjian Li
  - Zelong Zhang
  - Xiao Feng
  - Feng Zhang
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2604.06185
doi:
url: https://arxiv.org/abs/2604.06185
citation_key: yu2026benchmarking
paper_type: benchmark

read_status: unread

domain: agents
---

## Summary

This paper introduces WildToolBench, a benchmark for multi-turn, multi-step LLM tool use built from real user behavior patterns rather than idealized synthetic tasks. The benchmark targets three failure modes that the authors argue are central in practice: [[compositional-task]]s requiring nontrivial [[tool-orchestration]], [[hidden-intention]] distributed across dialogue history, and frequent [[instruction-transition]]s between tool-using and tool-free turns. The dataset contains 256 scenarios with 1,024 tasks, uses 400 tool lists covering about 1,600 APIs, and evaluates 57 proprietary and open-source models via exact tool-trajectory matching. Results are stark: no system exceeds `14.45%` session accuracy, and even frontier models remain brittle on mixed sequential-parallel topologies and long-range contextual inference.

## Problem & Motivation

Existing tool-use benchmarks were becoming saturated while still missing the behavioral messiness of real users. The paper argues that benchmark difficulty had been driven mainly by synthetic multi-step complexity, not by realistic dialogue phenomena. From large-scale user-log analysis, the authors isolate three recurring issues: users often ask naturally phrased but [[compositional-task]] requests, leave crucial information implicit and force [[hidden-intention]] recovery across turns, and switch rapidly between chat, clarification, single-call, and multi-call modes. The motivation behind WildToolBench is therefore to evaluate whether an LLM can behave as a robust agent in context, not merely emit valid function-call syntax.

## Method

- **Dialogue formalization**: a session is modeled as `D = {u_1, a_1, u_2, a_2, ..., u_N, a_N}`, with scattered user tasks `\{g_1, ..., g_M\}`. For a task `g_j`, tool interaction is `T^j = {a_1^T, e_1, a_2^T, e_2, ..., a_S^T, e_S}`, where `a^T` is a tool call and `e` is environment feedback.
- **Policy space**: WildToolBench explicitly mixes turns with `S = 0` (chat or clarification), `S = 1` (single-tool invocation), and `S > 1` (multi-step tool use). The dialogue is framed as an MDP whose state is the full history `{u, a, a^T, e}` and whose actions are response tokens that implement different policies.
- **Challenge taxonomy**: the benchmark is built around three axes: [[tool-orchestration]] for compositional tasks, [[hidden-intention]] inference across dialogue context, and [[instruction-transition]] between task types. Tool topology is further separated into sequential `g_multi^S`, parallel `g_multi^P`, and mixed `g_multi^{S+P}` execution structures.
- **Data curation pipeline**: the authors first mine real user logs for seed scenarios and few-shot behavior patterns, then collect `>1600` public APIs and retain `400` tool lists, and finally use a [[multi-agent-system]] to generate candidate trajectories that humans verify and annotate as ground truth. The final release contains `256` scenarios with `4` tasks each, i.e. `1024` tasks total.
- **Evaluation protocol**: the benchmark uses an enumerate-match-score procedure for valid tool trajectories instead of a single reference path. It reports task accuracy, session accuracy (all four tasks in a scenario solved), acceptable-path rate (AP), and optimal-path rate (OP), making execution efficiency part of evaluation rather than only terminal correctness.
- **Inference settings**: proprietary models are run with native function calling and default official hyperparameters; open-source models use official APIs when available or Hugging Face `4.51.0`, with default generation settings except `max_new_tokens = 512`.

## Key Results

- Across `57` evaluated models, no system exceeds `14.45%` session accuracy; the best task accuracy in the full table is `61.04%` for Gemini-2.0-Thinking.
- Strong open-source models remain competitive but behind the best proprietary systems: GLM-4.5 reaches `56.05%` task accuracy and `12.11%` session accuracy, while Kimi-K2 reaches `53.71%` and `10.55%`.
- On [[tool-orchestration]], the best overall multi-step task accuracy is only `43.75%` and the best OP rate is `42.74%`, both achieved by Claude-4-Sonnet; mixed sequential-plus-parallel tasks peak at just `25.00%`.
- For hidden-intent evaluation, long-range dependency tasks are the hardest subcase: no model surpasses `50%` accuracy, and the spread between models reaches `17.3` points.
- Frequent [[instruction-transition]]s consistently reduce accuracy; the paper reports drops of up to `30%` as transition count increases within a four-task dialogue.
- Error analysis shows a shift from syntax failures to reasoning failures: Gemini-2.0-Thinking has a `24.56%` refusal rate, Grok-4 has `24.07%` wrong-tool-name errors, and specialized models such as xLAM-2-70B and Watt-8B exceed `30%` on wrong-name errors.

## Limitations

- The benchmark relies on human annotation and verification for data quality, which constrains how far the dataset can scale.
- Maintaining both coverage of policy-transition types and annotation quality limits task length, so WildToolBench does not fully explore very long-horizon agent trajectories.
- The benchmark is grounded in real-user behavior patterns, but the final dataset is still curated rather than collected from a fully live deployment loop.
- The benchmark exposes severe weaknesses in current models, but it does not itself solve the training-data or system-design problems needed to close the gap.

## Concepts Extracted

- [[tool-use]]
- [[tool-orchestration]]
- [[compositional-task]]
- [[hidden-intention]]
- [[instruction-transition]]
- [[multi-turn-dialogue]]
- [[benchmark-dataset]]
- [[benchmark-evaluation]]
- [[multi-agent-system]]
- [[long-range-dependency]]

## Entities Extracted

- [[peijie-yu]]
- [[wei-liu]]
- [[yifan-yang]]
- [[jinjian-li]]
- [[zelong-zhang]]
- [[xiao-feng]]
- [[feng-zhang]]
- [[wildtoolbench]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
