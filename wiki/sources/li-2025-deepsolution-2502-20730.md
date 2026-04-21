---
type: source
subtype: paper
title: "DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking"
slug: li-2025-deepsolution-2502-20730
date: 2026-04-20
language: en
tags: [rag, benchmark, engineering-design, tree-search, llm]
processed: true

raw_file: raw/papers/li-2025-deepsolution-2502-20730/paper.pdf
raw_md: raw/papers/li-2025-deepsolution-2502-20730/paper.md
bibtex_file: raw/papers/li-2025-deepsolution-2502-20730/paper.bib
possibly_outdated: false

authors:
  - Zhuoqun Li
  - Haiyang Yu
  - Xuanang Chen
  - Hongyu Lin
  - Yaojie Lu
  - Fei Huang
  - Xianpei Han
  - Yongbin Li
  - Le Sun
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2502.20730
doi: 10.48550/arXiv.2502.20730
url: http://arxiv.org/abs/2502.20730
citation_key: li2025deepsolution
paper_type: method

read_status: unread
domain: retrieval
---

## Summary

This paper targets complex engineering solution design, a RAG setting where inputs contain multiple real-world constraints and outputs must be complete, feasible plans rather than short answers. It contributes both [[solutionbench]], a benchmark built from authoritative engineering reports across eight domains, and [[solutionrag]], an inference-time system that alternates solution drafting and review inside a bi-point thinking tree. The method retrieves domain knowledge for multiple sampled improvement directions, scores both candidate solutions and review comments with logit-based node evaluation, and prunes the search to the most promising branch. On SolutionBench, SolutionRAG consistently outperforms deep-reasoning-only baselines and prior single-round or multi-round RAG systems, reaching overall analytical and technical scores of `66.2` and `64.1`.

## Problem & Motivation

Prior [[retrieval-augmented-generation]] work mainly studies multi-hop QA and long-form QA, where outputs are entity answers or integrated paragraphs. The authors argue that complex engineering requirements are harder: they contain multiple interacting real-world constraints and demand reliable end-to-end solutions. Pure LLM reasoning lacks enough domain knowledge, while standard RAG pipelines do not explicitly support iterative design, critique, and repair. The paper therefore defines a new task, constructs a benchmark grounded in real engineering reports, and proposes a tree-search-style RAG system that can iteratively improve draft solutions while checking whether all constraints are satisfied.

## Method

- **Benchmark construction**: build [[solutionbench]] from authoritative engineering journals in `8` domains: environment, mining, transportation, aerospace, telecom, architecture, water resource, and farming.
- **Extraction pipeline**: convert PDFs to text with Marker, then use `GPT-4o` plus a manual template to extract `Requirement`, `Solution`, `Analytical Knowledge`, `Technical Knowledge`, and `Explanation`; manually verify correctness and merge duplicate knowledge within each domain.
- **Dataset formalization**: each domain dataset is written as `` `\mathcal{D} = {q_i, s_i, {k_j^(a)}_{j=1}^{A_i}, {k_j^(t)}_{j=1}^{T_i}, e_i}_{i=1}^N` ``, while the domain knowledge base is `` `\mathcal{K} = \cup[{k_j^(a)}_{j=1}^{A_i}, {k_j^(t)}_{j=1}^{T_i}] = {k_i}_{i=1}^M` ``.
- **Task setting**: the main evaluation target is RAG inference, formulated as `` `\hat{s} = \mathcal{F}(q, \mathcal{K})` ``, where a system must generate a reliable solution from a requirement and a domain knowledge base.
- **Bi-point thinking tree**: [[solutionrag]] alternates solution nodes and comment nodes, with `` `s_j^(i) -> {c_h^(i+1)}_{h=1}^H` `` and `` `c_j^(i+1) -> {s_h^(i+2)}_{h=1}^H` ``. This operationalizes [[tree-based-exploration]] plus [[bi-point-thinking]].
- **Design expansion**: sample `H` design proposals with `` `{p_h}_{h=1}^H = \text{LLM}(q, c_j^(i+1))` ``, retrieve proposal-specific knowledge with `` `\mathcal{K}_h = \text{Retrieval}(p_h, \mathcal{K}) = {k_r}_{r=1}^R` ``, then generate refined solutions by `` `s_h^(i+2) = \text{LLM}(q, s_z^(i), c_j^(i+1), \mathcal{K}_h)` ``.
- **Review expansion**: for each solution node, generate `H` review directions, retrieve supporting evidence, and produce comment nodes that identify deficiencies relative to the original requirement.
- **Node evaluation and pruning**: score solution reliability with `` `\mathcal{J}_h(s_j^(i)) = \text{Logits}(u_s | s_j^(i), c_h^(i+1))` `` where `u_s` is “According to the comment, above solution is reliable”; score comment helpfulness with `` `\mathcal{J}_h(c_j^(i+1)) = \text{Logits}(u_c | s_z^(i), c_j^(i+1), s_h^(i+2))` `` where `u_c` is “Comparing the new solution and old solution, the comment is helpful”; average across children and keep the top-`W` nodes per layer.
- **Implementation details**: set maximum depth `L = 5`, children per node `H = 2`, retained width `W = 1`, retrieval depth `R = 10`; use `Qwen2.5-7B-Instruct` as the base generator, `NV-Embed-v2` as retriever, `GPT-4o` as the evaluator for analytical and technical scores, and serve RAG baselines plus SolutionRAG through `vLLM`.

## Key Results

- [[solutionbench]] contains `950` datapoints and `6,137` knowledge entries across `8` engineering domains; per-domain datapoints are `119/117/124/115/116/118/119/122`.
- On the main benchmark table, [[solutionrag]] reaches analytical / technical scores of `66.4/67.9` (environment), `59.7/50.5` (mining), `64.1/58.5` (transportation), `69.9/72.7` (aerospace), `68.8/69.0` (telecom), `67.9/68.0` (architecture), `66.0/60.7` (water resource), and `66.9/65.2` (farming).
- The ablation table reports overall scores of `66.2/64.1` for full SolutionRAG, versus `62.7/61.7` without tree structure and `62.9/61.5` without bi-point thinking.
- In mining, SolutionRAG improves technical score by `+10.4` over Naive-RAG (`50.5` vs `40.1`) and by `+8.9` over Self-RAG (`50.5` vs `41.6`).
- Deep reasoning models without retrieval remain weaker on this task; for example, `o1-2024-12-17` scores only `37.5` technical points in mining, and `GLM-Zero-Preview` reaches only `42.3` analytical points in aerospace.

## Limitations

- The evaluation relies on `GPT-4o` as an automatic judge for both analytical and technical scores, so metric quality depends on the evaluator prompt and reference packaging rather than direct expert scoring.
- Benchmark construction uses LLM-assisted extraction and translation before manual verification; extraction bias or template bias can still shape what knowledge is retained in the final benchmark.
- The method is purely inference-time and does not include specialized training; the authors explicitly note that reinforcement learning or other training could yield stronger systems.
- Tree hyperparameters are only lightly explored because of limited GPU resources; the paper fixes `L = 5`, `H = 2`, and `W = 1`, so the width-depth-performance tradeoff remains under-studied.

## Concepts Extracted

- [[complex-engineering-solution-design]]
- [[retrieval-augmented-generation]]
- [[benchmark-dataset]]
- [[dataset-construction]]
- [[iterative-refinement]]
- [[tree-based-exploration]]
- [[bi-point-thinking]]
- [[node-evaluation]]
- [[analytical-knowledge]]
- [[technical-knowledge]]

## Entities Extracted

- [[zhuoqun-li]]
- [[haiyang-yu]]
- [[xuanang-chen]]
- [[hongyu-lin]]
- [[yaojie-lu]]
- [[fei-huang]]
- [[xianpei-han]]
- [[yongbin-li]]
- [[le-sun]]
- [[tongyi-lab]]
- [[gpt-4o]]
- [[solutionbench]]
- [[solutionrag]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
