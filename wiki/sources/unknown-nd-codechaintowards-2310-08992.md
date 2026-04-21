---
type: source
subtype: paper
title: "CODECHAIN: TOWARDS MODULAR CODE GENERATION THROUGH CHAIN OF SELF-REVISIONS WITH REPRESENTATIVE SUB-MODELS"
slug: unknown-nd-codechaintowards-2310-08992
date: 2026-04-20
language: en
tags: [llm, code-generation, modularity, self-revision, program-synthesis]
processed: true
raw_file: raw/papers/unknown-nd-codechaintowards-2310-08992/paper.pdf
raw_md: raw/papers/unknown-nd-codechaintowards-2310-08992/paper.md
bibtex_file: raw/papers/unknown-nd-codechaintowards-2310-08992/paper.bib
possibly_outdated: true
authors:
  - Hung Le
  - Hailin Chen
  - Amrita Saha
  - Akash Gokul
  - Doyen Sahoo
  - Shafiq Joty
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.08992
doi:
url: https://arxiv.org/abs/2310.08992
citation_key: unknownndcodechaintowards
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

CodeChain is an inference-time framework for code generation that tries to turn large language models into more modular programmers. Instead of sampling monolithic programs independently, it first prompts the model to outline reusable sub-modules with chain-of-thought prompting, then iteratively revises solutions by extracting sub-modules from previous samples, clustering them, and feeding representative cluster centroids back into later rounds. The paper evaluates this procedure on APPS and CodeContests with GPT-3.5, GPT-4, and WizardCoder. Across both benchmarks, the method improves pass@1, especially on harder competitive-programming tasks, and also increases the modularity and reusability of generated code. The central claim is that representative sub-module reuse is a more effective revision signal than treating each candidate program independently.

## Problem & Motivation

The paper targets a gap between current LLM code generation and human programming practice. Strong models already solve simpler tasks from benchmarks like HumanEval or MBPP, but they struggle on harder programming contests where decomposition, abstraction, and iterative debugging matter. The authors argue that common sampling-based methods generate whole solutions independently and therefore waste collective structure that appears across multiple partially correct candidates. CodeChain is motivated by an experienced developer workflow: decompose the task into sub-problems, reuse promising modules, and revise the solution over multiple attempts instead of restarting from scratch each time.

## Method

- The task is framed as sequence-to-sequence code generation: given a problem description `D`, generate a solution program `W = (w_1, ..., w_T)` and evaluate it by whether it passes hidden test cases.
- CodeChain starts with chain-of-thought prompting that explicitly asks the model to outline sub-modules `S_i` as function headers plus docstrings before producing the final code, rather than directly emitting one monolithic block.
- Initial modular generation factorizes into sub-module generation and final-token generation: `S_i ~ p_theta(. | S_{1:i-1}, D)` and `w_t ~ p_theta(. | w_{1:t-1}, {S_i}, D)`.
- Given a budget of `N = 20` samples per round, the method extracts all sub-modules across samples, embeds them, and applies `K`-means clustering to group semantically similar implementations.
- For each cluster `k`, it selects a representative centroid module `C_k = arg min ||S_i^k - mu_k||`, where `mu_k` is the embedding centroid. These representatives are treated as reusable code hints.
- In revision round `R`, generation is conditioned on prior centroid modules: `w_t^R ~ p_theta(. | w_{1:t-1}^R, {S_i^R}, C^{R-1}, D)` and `S_i^R ~ p_theta(. | S_{1:i-1}^R, C^{R-1}, D)`.
- When public tests are available, the method filters failed programs before clustering, so the feedback pool is biased toward more plausible implementations.
- Experiments run up to `5` self-revision rounds; the best round is selected by validation performance, and APPS often peaks at round `4`.
- Base models include GPT-3.5, GPT-4, and WizardCoder (`1B` to `34B`). Generation uses temperature `0.6`, maximum output length `2048`, and StarEncoder embeddings for sub-module clustering.

## Key Results

- On APPS test pass@1, CodeChain lifts GPT-3.5 from `22.33` to `30.24` and WizardCoder-15B from `7.90` to `10.50`.
- On APPS with CodeT-style filtering, GPT-3.5 improves from `32.54` to `35.34` average pass@1; competition-level pass@1 rises from `9.46` to `15.08`.
- On the 20-problem APPS subset used to compare with Self-repair, GPT-4 improves from `34.75` to `61.50` average pass@1, exceeding Self-repair+GPT-4 with human feedback at `52.60`.
- On CodeContests test pass@1, GPT-3.5 improves from `5.82` to `10.27` without filtering and from `11.34` to `13.75` with CodeT filtering; WizardCoder-15B improves from `1.98` to `2.48`.
- In appendix analysis, APPS gains typically peak at revision round `4`, with reported average improvements over direct generation of roughly `1.6x` for GPT-3.5 and `2x` for WizardCoder; competition-level gains exceed `2x` and `5x`, respectively.
- Qualitative evaluation with GPT-4 judges shows CodeChain outputs are much more modular and reusable; the paper reports most CodeChain samples scoring `3` to `5`, while direct generation produces non-modular code about `80%` of the time.

## Limitations

- Chain-of-thought prompting alone can reduce correctness before self-revision, so the method depends on later revision rounds to recover accuracy.
- The approach is sensitive to feedback quality: filtering with imperfect synthetic tests is notably worse than using public or private tests.
- Repeated revision can overfit to sparse public tests; on APPS, performance drops slightly by round `5`.
- The pipeline increases inference cost because it requires multiple sampled programs, sub-module extraction, embedding, clustering, and repeated regeneration.
- Evaluation is concentrated on competitive-programming benchmarks and Python-centric code generation, so generalization to broader software-engineering settings is not established.

## Concepts Extracted

- [[large-language-model]]
- [[code-generation]]
- [[sequence-to-sequence]]
- [[chain-of-thought-prompting]]
- [[modular-code-generation]]
- [[self-revision]]
- [[semantic-clustering]]
- [[code-reuse]]
- [[execution-feedback]]
- [[pass-at-k]]
- [[unit-test-based-evaluation]]

## Entities Extracted

- [[hung-le]]
- [[hailin-chen]]
- [[amrita-saha]]
- [[akash-gokul]]
- [[doyen-sahoo]]
- [[shafiq-joty]]
- [[salesforce-research]]
- [[apps-benchmark]]
- [[codecontests]]
- [[gpt-3-5]]
- [[gpt-4]]
- [[wizardcoder]]
- [[starcoder]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
