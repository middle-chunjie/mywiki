---
type: source
subtype: paper
title: "EvoCodeBench: An Evolving Code Generation Benchmark with Domain-Specific Evaluations"
slug: li-2024-evocodebench-2410-22821
date: 2026-04-20
language: en
tags: [benchmark, code-generation, evaluation, llm, software-engineering]
processed: true

raw_file: raw/papers/li-2024-evocodebench-2410-22821/paper.pdf
raw_md: raw/papers/li-2024-evocodebench-2410-22821/paper.md
bibtex_file: raw/papers/li-2024-evocodebench-2410-22821/paper.bib
possibly_outdated: false

authors:
  - Jia Li
  - Ge Li
  - Xuanming Zhang
  - Yunfei Zhao
  - Yihong Dong
  - Zhi Jin
  - Binhua Li
  - Fei Huang
  - Yongbin Li
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2410.22821
doi:
url: http://arxiv.org/abs/2410.22821
citation_key: li2024evocodebench
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

EvoCodeBench introduces an evolving benchmark for repo-level Python code generation that is designed to reduce data leakage and expose domain-specific strengths and weaknesses of large language models. The first release, EvoCodeBench-2403, contains `275` tasks from `25` recent repositories, each annotated with requirements, repository context, reference code, dependencies, domain labels, and tests. The benchmark evaluates both functional correctness and dependency usage through `Pass@k` and `Recall@k`, and adds Domain-Specific Improvement to characterize comfort and strange domains. Across `8` popular LLMs, the paper shows substantially lower scores than older benchmarks, suggesting harder and less contaminated evaluation, while also revealing cross-domain performance heterogeneity that aggregate scores obscure.

## Problem & Motivation

Existing code-generation benchmarks suffer from two linked problems. First, many benchmarks are old enough that contemporary code LLMs may already have seen their test data during training, which makes performance numbers hard to trust. Second, most benchmarks report only aggregate scores and therefore fail to tell practitioners which model is actually strong for a particular programming domain such as databases or Internet software. The paper aims to build a repo-level benchmark from recent repositories, preserve realistic dependency structure, and add domain labels plus domain-aware analyses so that evaluation is both more trustworthy and more actionable.

## Method

- **Benchmark scope**: releases `EvoCodeBench-2403` with `275` Python tasks from `25` repositories; each sample contains `7` fields: signature, requirement, repository, reference code, reference dependencies, domain label, and test cases.
- **Stage I, repository selection**: crawls GitHub repositories created within the last `6` months, requiring permissive licenses, non-fork, non-malicious status, `> 50` stars, and explicit unit tests; trivial functions are removed during scraping.
- **Stage II, execution-based filtering**: installs environments with `pip` and runs tests with `pytest`; functions without executable tests are discarded so each retained task has runnable functional validation.
- **Stage III, automatic annotation**: uses a static-analysis parser to extract signatures, reference code, and dependencies, then uses `gpt-4` one-shot prompting to generate natural-language requirements and domain labels under a manually designed `10`-domain taxonomy derived from PyPI statistics.
- **Stage IV, benchmark construction**: samples candidate functions to match real-world repository statistics, targeting code-distribution parity (`27%` standalone / `73%` non-standalone) and dependency-distribution parity (`3.46` average dependencies vs `3.22` in `500` real repositories).
- **Core metrics**: functional correctness uses `Pass@k = E[1 - C(n-c, k) / C(n, k)]`; dependency usage uses `Recall@k = E[max_i |R ∩ P_i| / |R|]`; domain specialization uses `DSI_i = (1 / (N - 1)) Σ_j ((P_i - P_j) / P_i) * 100`.
- **Experimental protocol**: evaluates `8` LLMs under without-context, local-file completion, and local-file infilling settings with `k ∈ {1, 3, 5, 10}`; uses greedy decoding for `k = 1`, otherwise nucleus sampling with `20` samples, `temperature = 0.4`, `top-p = 0.95`, and `max_len = 500`.
- **Domain analysis and RAG**: defines comfort/strange domains with threshold `T = 10%`; also tests a simple retrieval-augmented setup that retrieves top-`5` functions with similar names from the current repository.

## Key Results

- **Leakage drops sharply**: CDD estimates HumanEval leakage at `41.47%` for `gpt-3.5`, while EvoCodeBench-2403 is between `0.73%` and `2.18%` across tested models (`2.18%` for `gpt-4`, `1.75%` for `gpt-3.5`).
- **Repo-level coding is much harder than older benchmarks**: in local-file infilling, `gpt-4` reaches only `20.73` Pass@1 and `68.24` Recall@1; the best Pass@10 is `26.01` from DeepSeek Coder `33B`, while the best Recall@10 is `86.25`, also from DeepSeek Coder `33B`.
- **Context helps substantially**: relative to no-context generation, `gpt-4` Pass@1 rises from `7.27` to `17.45` in local-file completion and to `20.73` in infilling, showing that repository-local context is a major contributor to performance.
- **Simple RAG is already useful**: for `gpt-4`, similar-function retrieval raises Pass@1 from `8.31` to `12.29` and Recall@1 from `21.08` to `45.14`; for `gpt-3.5`, Pass@1 rises from `6.64` to `11.62`.
- **Domain-level rankings differ from overall rankings**: `gpt-4` leads most domains but scores only `20.00` in Internet, below several peers at `26.67`; StarCoder 2 `15B` reaches `38.89` Pass@1 on Database, matching `gpt-4` and outperforming larger code models there.
- **Annotation quality is close to human work at much lower cost**: auto-generated requirements are acceptable in `96.7%` of cases and domain labels in `98.5%`; `gpt-4` annotation costs `1h9m` and `$3.11` versus `23h` and `$172.5` for humans.

## Limitations

- The benchmark is Python-only, so its conclusions do not directly transfer to multilingual code generation or languages with different build and testing constraints.
- The first release is relatively small (`275` samples) compared with some existing benchmarks because it intentionally restricts itself to recent repositories (`Oct 2023` to `Mar 2024`).
- Domain coverage is still imbalanced in version `2403`; some domains have very few samples and were excluded from fine-grained analysis when counts were below `10`.
- Requirement and domain annotations are partly LLM-generated, and the paper notes residual failure modes such as missing requirement details or incorrect domain labels for API-specific cases.

## Concepts Extracted

- [[large-language-model]]
- [[code-generation-benchmark]]
- [[repo-level-code-generation]]
- [[data-leakage]]
- [[domain-specific-evaluation]]
- [[pass-at-k]]
- [[recall-at-k]]
- [[domain-taxonomy]]
- [[static-analysis]]
- [[retrieval-augmented-generation]]

## Entities Extracted

- [[jia-li]]
- [[ge-li]]
- [[xuanming-zhang]]
- [[yunfei-zhao]]
- [[yihong-dong]]
- [[zhi-jin]]
- [[binhua-li]]
- [[fei-huang]]
- [[yongbin-li]]
- [[peking-university]]
- [[bytedance]]
- [[alibaba-group]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
