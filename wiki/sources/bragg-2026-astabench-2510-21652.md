---
type: source
subtype: paper
title: "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite"
slug: bragg-2026-astabench-2510-21652
date: 2026-04-20
language: en
tags: [agents, benchmarking, scientific-discovery, evaluation, literature-search]
processed: true

raw_file: raw/papers/bragg-2026-astabench-2510-21652/paper.pdf
raw_md: raw/papers/bragg-2026-astabench-2510-21652/paper.md
bibtex_file: raw/papers/bragg-2026-astabench-2510-21652/paper.bib
possibly_outdated: false

authors:
  - Jonathan Bragg
  - Mike D'Arcy
  - Nishant Balepur
  - Dan Bareket
  - Bhavana Dalvi
  - Sergey Feldman
  - Dany Haddad
  - Jena D. Hwang
  - Peter Jansen
  - Varsha Kishore
  - Bodhisattwa Prasad Majumder
  - Aakanksha Naik
  - Sigal Rahamimov
  - Kyle Richardson
  - Amanpreet Singh
  - Harshit Surana
  - Aryeh Tiktinsky
  - Rosni Vasu
  - Guy Wiener
  - Chloe Anastasiades
  - Stefan Candra
  - Jason Dunkelberger
  - Dan Emery
  - Rob Evans
  - Malachi Hamada
  - Regan Huff
  - Rodney Kinney
  - Matt Latzke
  - Jaron Lochner
  - Ruben Lozano-Aguilera
  - Cecile Nguyen
  - Smita Rao
  - Amber Tanaka
  - Brooke Vlahos
  - Peter Clark
  - Doug Downey
  - Yoav Goldberg
  - Ashish Sabharwal
  - Daniel S. Weld
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2510.21652
doi:
url: https://arxiv.org/abs/2510.21652
citation_key: bragg2026astabench
paper_type: benchmark

read_status: unread
read_date:
rating:

domain: agents
---

## Summary

AstaBench introduces a controlled benchmark suite for scientific-research agents that pairs `11` tasks and `2400+` problems with standardized tools, cost accounting, and baseline agents. The suite spans literature search, long-form QA, literature-table generation, code execution, data-driven discovery, and end-to-end research workflows. Its main technical contribution is not a single model, but an evaluation stack: date-restricted scientific-corpus search, a stateful notebook environment, an Inspect-based leaderboard layer with normalized cost reporting, and a broad baseline suite covering `57` agents from `22` architectural classes. The reported results show that specialized tooling materially improves performance, but also that current agents remain far from robust scientific assistance, especially for coding, data analysis, and full research execution.

## Problem & Motivation

Prior agent benchmarks do not adequately measure realistic scientific research assistance. Existing suites are often narrow in task scope, weakly connected to real product usage, missing standardized tools for controlled comparison, and inconsistent about confounders such as model cost or privileged tool access. The paper argues that evaluating science agents requires holistic coverage of the research pipeline plus reproducible environments, because otherwise improvements may reflect better search access or more spending rather than stronger agentic reasoning.

## Method

- **Suite design**: AstaBench packages `11` benchmarks across `4` categories: Literature Understanding, Code & Execution, Data Analysis, and End-to-End Discovery, with examples such as PaperFindingBench (`267` test / `66` val), ScholarQA-CS2 (`100` / `100`), SUPER-Expert (`45` / `50`), DiscoveryBench (`239` / `25`), and E2E-Bench (`40` / `10`).
- **Standard tool environment**: each task exposes a subset of Asta Environment tools, primarily Asta Scientific Corpus and a stateful Computational Notebook. The corpus supports date-restricted retrieval so benchmark inputs can be evaluated against a frozen literature state instead of the live web.
- **Corpus APIs**: the benchmark exposes MCP tools such as `snippet_search`, `search_papers_by_relevance`, `get_paper`, `get_paper_batch`, `get_citations`, `search_authors_by_name`, `get_author_papers`, and `search_paper_by_title`.
- **Notebook execution model**: the notebook preserves Python state across calls, supports IPython magics and shell commands, and times out a single cell after `5` minutes. This allows both direct tool-calling agents and code-writing agents to operate inside the same evaluation framework.
- **Cost normalization**: the `agent-eval` layer computes normalized dollar cost from Inspect usage logs using a frozen LiteLLM price snapshot. It includes cache discounts, excludes latency-tier discounts, and reports Pareto tradeoffs between score and cost.
- **Confounder tracking**: leaderboard submissions are labeled by openness and tooling class, e.g. open-source/open-weight vs closed/API and `standard` vs `custom interface` vs `fully custom`, so comparisons are not reduced to one scalar score.
- **PaperFindingBench evaluation**: navigational and metadata queries use set-level `F1`; semantic queries use the harmonic mean of `nDCG` and estimated `recall@k`, where `k` is an estimated relevant-set size derived from lenient retrieval unions.
- **ScholarQA-CS2 evaluation**: long-form QA is scored by averaging `4` judge-based facets: citation recall, citation precision, answer relevance, and answer coverage. Coverage is computed from clustered rubric ingredients gathered from candidate system outputs.
- **Table and discovery evaluation**: ArxivDIGESTables-Clean unrolls tables into atomic statements and measures entailment recall against a reference table; DiscoveryBench scores hypothesis alignment over `context`, `variables`, and `relationship`.
- **End-to-end evaluation**: E2E-Bench tasks specify a full AI/NLP research workflow, and LLM judges score report, code, and artifacts against rubric items. Because these agents are expensive, the framework also supports cache-based evaluation.
- **Baseline coverage**: the released agents include `9` science-optimized Asta classes plus general baselines such as ReAct and Smolagents Coder, yielding experiments over `57` agents in `22` classes.

## Key Results

- Overall, **Asta v0** is best among full-suite agents at `53.0` score and `$3.40` per problem, beating **ReAct + gpt-5** at `44.0` and `$0.31`.
- Literature search remains difficult: **Asta Paper Finder** reaches `39.7 ± 3.1` on PaperFindingBench and `90.7 ± 6.6` on LitQA2-FullText-Search.
- Long-form literature QA is comparatively strongest: **Asta Scholar QA (w/ Tables)** scores `87.9 ± 1.2` on ScholarQA-CS2, while **Elicit** scores `85.5 ± 1.6` and **SciSpace Deep Review** `84.6 ± 1.3`.
- Literature table generation is still weak: **Asta Table Synthesis + gpt-5** scores `42.6 ± 3.5`; the `gpt-5-mini` variant gets `41.7 ± 3.7` at only about `13%` of the cost.
- Code and execution are major bottlenecks: on SUPER-Expert, only **ReAct + gpt-5** breaks `40`, reaching `41.1 ± 12.9`; most agents remain below `25`.
- Data-driven discovery is also unsolved: the best DiscoveryBench score is `33.7 ± 5.1` from **ReAct + o3**.
- End-to-end task scores can look moderate without implying full success: **Asta Panda + claude-sonnet-4** reaches `70.5 ± 6.2` on E2E-Bench and `68.2 ± 4.4` on E2E-Bench-Hard, yet the paper estimates only about `1%` complete success on full multi-step experiments.
- Among open-source agents with open weights, the best reported overall score is only `11.1` from **Smolagents Coder + Llama-4-Scout**, underscoring the current gap to closed-model systems.

## Limitations

- The suite is broad but still weighted toward computer-science workflows, especially in literature and end-to-end discovery tasks.
- Some evaluations depend on LLM-as-a-judge procedures and prompt design, which introduces evaluator sensitivity even when correlations are reported as strong.
- Certain leaderboard entries rely on cached or API-only systems rather than fully reproducible open implementations.
- E2E-Bench gives fairly detailed task instructions, so it only weakly measures open-ended ideation and planning compared with unconstrained scientific discovery.
- CORE-Bench-Hard removes GPU-required capsules, improving accessibility but narrowing the reproduction setting relative to the full difficulty spectrum.

## Concepts Extracted

- [[benchmark]]
- [[large-language-model]]
- [[tool-augmented-agent]]
- [[scientific-literature-search]]
- [[literature-understanding]]
- [[cost-aware-evaluation]]
- [[llm-as-a-judge]]
- [[experiment-reproduction]]
- [[data-driven-discovery]]
- [[end-to-end-discovery]]

## Entities Extracted

- [[jonathan-bragg]]
- [[mike-darcy]]
- [[nishant-balepur]]
- [[dan-bareket]]
- [[bhavana-dalvi]]
- [[sergey-feldman]]
- [[dany-haddad]]
- [[jena-d-hwang]]
- [[peter-jansen]]
- [[varsha-kishore]]
- [[bodhisattwa-prasad-majumder]]
- [[aakanksha-naik]]
- [[sigal-rahamimov]]
- [[kyle-richardson]]
- [[amanpreet-singh]]
- [[harshit-surana]]
- [[aryeh-tiktinsky]]
- [[rosni-vasu]]
- [[guy-wiener]]
- [[chloe-anastasiades]]
- [[stefan-candra]]
- [[jason-dunkelberger]]
- [[dan-emery]]
- [[rob-evans]]
- [[malachi-hamada]]
- [[regan-huff]]
- [[rodney-kinney]]
- [[matt-latzke]]
- [[jaron-lochner]]
- [[ruben-lozano-aguilera]]
- [[cecile-nguyen]]
- [[smita-rao]]
- [[amber-tanaka]]
- [[brooke-vlahos]]
- [[peter-clark]]
- [[doug-downey]]
- [[yoav-goldberg]]
- [[ashish-sabharwal]]
- [[daniel-s-weld]]
- [[allen-institute-for-ai]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
