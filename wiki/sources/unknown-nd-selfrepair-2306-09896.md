---
type: source
subtype: paper
title: Is Self-Repair a Silver Bullet for Code Generation?
slug: unknown-nd-selfrepair-2306-09896
date: 2026-04-20
language: en
tags: [llm, code-generation, self-repair, debugging, evaluation]
processed: true

raw_file: raw/papers/unknown-nd-selfrepair-2306-09896/paper.pdf
raw_md: raw/papers/unknown-nd-selfrepair-2306-09896/paper.md
bibtex_file: raw/papers/unknown-nd-selfrepair-2306-09896/paper.bib
possibly_outdated: true

authors:
  - Theo X. Olausson
  - Jeevana Priya Inala
  - Chenglong Wang
  - Jianfeng Gao
  - Armando Solar-Lezama
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2306.09896
doi:
url: https://arxiv.org/abs/2306.09896
citation_key: unknownndselfrepair
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper studies whether self-repair is genuinely better than simply drawing more independent code samples under the same compute budget. It formalizes a repair pipeline with code generation, execution, feedback generation, and repair, then evaluates CodeLlama-13B-Instruct, GPT-3.5, and GPT-4 on HumanEval and a 300-task APPS subset. The central finding is negative but nuanced: self-repair often yields only modest gains, can lose to i.i.d. resampling at small budgets, and benefits more from diverse initial programs than from many repair branches. The paper further shows that the main bottleneck is feedback quality: stronger external feedback models consistently improve repair, and human-written feedback improves GPT-4's repair success far beyond GPT-4's self-generated explanations.

## Problem & Motivation

The paper asks a budget-sensitive question that earlier self-repair studies largely avoid: if repair requires extra model calls, is it actually better than spending the same budget on additional i.i.d. code samples? The authors argue that code-generation quality alone does not determine self-repair effectiveness. A second capability matters just as much: the model must diagnose why its own failing program is wrong. This motivates isolating the feedback stage and comparing true self-repair against two stronger alternatives, namely boosted feedback from a better model and feedback from human programmers.

## Method

- The framework decomposes self-repair into four stages: code generation, execution, feedback generation, and repair. Initial programs are sampled as `p_i ~iid M_P(psi)` for `i = 1..n_p`.
- Failed programs are executed against the full unit-test bed, producing error signals `e_i`; the paper assumes executable tests are fully available for each task.
- Feedback strings are sampled as `f_ij ~iid M_F(psi; p_i; e_i)` for `j = 1..n_f`, explicitly separating diagnosis from repair so that feedback quality can be ablated.
- Repairs are sampled as `r_ijk ~iid M_P(psi; p_i; e_i; f_ij)` for `k = 1..n_r`. When the same model handles both diagnosis and repair, the paper jointly samples `(f_ij, r_ij) ~iid M_P(psi; p_i; e_i)` with `n_fr` joint draws.
- The full search object is a repair tree `T` rooted at the specification, with effective program budget `|programs(T)| = n_p + n_p n_fr` in the joint case or `|programs(T)| = n_p + n_p n_f n_r` when feedback and repair are separated.
- Main evaluation compares self-repair against an equal-budget no-repair baseline using pass@k-style success. The paper also reports repair success rates and appendix variants based on sequential search and token-budget accounting.
- To make large sweeps tractable, the authors generate one large frozen tree per task and bootstrap subtrees with replacement. They use `N_p = 50`, `N_f = 25` in most experiments, `N_f = 10` in the boosted-feedback study, `N_r = n_r = 1`, and `N_t = 1000` bootstrap samples.
- Models are CodeLlama-13B-Instruct, GPT-3.5 (`gpt-3.5-turbo-0301`), and GPT-4 (`gpt-4-0314`). Tasks come from HumanEval and a 300-problem APPS subset with `180` interview, `60` competition, and `60` introductory problems. All decoding uses temperature `0.8`.
- The self-repair hyperparameter sweep uses `(n_p, n_fr) in {1,2,5,10,25} x {1,3,5,10}`. The stronger-feedback study uses `(n_p, n_f, n_r) in {1,...,25} x {1} x {1}`.
- The human-feedback study samples `40` failing GPT-4 programs, collects `2` human critiques per program from `16` participants, and draws `25` GPT-4 repair candidates for each `(program, feedback)` pair.

## Key Results

- Self-repair is not uniformly superior: for GPT-4 on APPS, allocating budget to diversity works better than allocating it to repeated repair. `n_p = 10, n_fr = 1` reaches `1.05x` the equal-budget baseline pass@20, while `n_p = 2, n_fr = 10` falls to `0.97x` of baseline pass@22.
- Relative gains are highly task- and model-dependent: GPT-4 on APPS improves by up to `8%`, GPT-3.5 on APPS competition problems by up to `34%`, CodeLlama on HumanEval by up to `10%`, and GPT-3.5 on HumanEval by only `3%`.
- Stronger feedback reliably helps. Appendix repair-success rates show GPT-3.5 on APPS rising from `4.7%` to `11.5%` when GPT-4 supplies feedback, and CodeLlama on HumanEval rising from `9.1%` to `39.3%` when GPT-4 supplies feedback.
- Human feedback outperforms model feedback by a wide margin for GPT-4 repair: overall repair success rises from `33.30%` with GPT-4's own explanations to `52.60%` with human explanations, a `1.58x` improvement.
- The human-feedback advantage is largest on harder tasks: GPT-4 repair success on competition problems increases from `3.67%` to `14.67%`, while introductory problems rise from `42.64%` to `62.21%`.
- Qualitative annotation over `80` human and `80` GPT-4 feedback snippets shows GPT-4 is much more often inaccurate (`32/80` vs. `7/80`), while humans are better at proposing high-level changes and occasionally expressing uncertainty.

## Limitations

- The bootstrap estimates are derived by sub-sampling from a single large frozen repair tree per task, which could introduce statistical artifacts even though the paper keeps `n_p` and repair branching well below the maximum tree size.
- The experiments cover only self-contained Python tasks with executable unit tests, so the conclusions may not transfer directly to real software engineering settings with partial specifications, long-range context, and sparse or missing tests.
- The human study evaluates feedback quality but not full human-in-the-loop cost: it does not measure debugging time, interaction overhead, or whether the stronger repair rate justifies the extra human effort.

## Concepts Extracted

- [[self-repair]]
- [[feedback-generation]]
- [[repair-tree]]
- [[code-generation]]
- [[program-repair]]
- [[code-execution]]
- [[unit-test-feedback]]
- [[pass-at-k]]
- [[cost-aware-evaluation]]
- [[human-feedback]]
- [[benchmark-dataset]]

## Entities Extracted

- [[theo-x-olausson]]
- [[jeevana-priya-inala]]
- [[chenglong-wang]]
- [[jianfeng-gao]]
- [[armando-solar-lezama]]
- [[mit-csail]]
- [[microsoft-research]]
- [[code-llama]]
- [[gpt-3-5]]
- [[gpt-4]]
- [[humaneval]]
- [[apps-benchmark]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
