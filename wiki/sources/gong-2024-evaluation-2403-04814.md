---
type: source
subtype: paper
title: Evaluation of LLMs on Syntax-Aware Code Fill-in-the-Middle Tasks
slug: gong-2024-evaluation-2403-04814
date: 2026-04-20
language: en
tags: [llm, code-generation, benchmark, evaluation, code-completion]
processed: true
raw_file: raw/papers/gong-2024-evaluation-2403-04814/paper.pdf
raw_md: raw/papers/gong-2024-evaluation-2403-04814/paper.md
bibtex_file: raw/papers/gong-2024-evaluation-2403-04814/paper.bib
possibly_outdated: false
authors:
  - Linyuan Gong
  - Sida Wang
  - Mostafa Elhoushi
  - Alvin Cheung
year: 2024
venue: ICML 2024
venue_type: conference
arxiv_id: 2403.04814
doi: 10.48550/arXiv.2403.04814
url: http://arxiv.org/abs/2403.04814
citation_key: gong2024evaluation
paper_type: benchmark
read_status: unread
domain: llm
---

## Summary

This paper introduces SAFIM, a syntax-aware fill-in-the-middle benchmark for evaluating code LLMs on realistic code-editing completions rather than standalone function synthesis. The benchmark contains `17,720` multilingual examples spanning algorithmic block completion, control-flow completion, and API function call completion, with sources restricted to code written after April 2022 to reduce contamination. The authors pair the dataset with five prompt designs and an AST-based syntax-aware truncation procedure that extracts valid completions from open-ended generations. Across `15` models, the study finds that FIM pretraining improves both infilling and standard left-to-right performance, while pretraining recipe and data quality matter more than raw parameter count. The work frames SAFIM as an evaluation platform for studying code-LLM pretraining choices under more realistic repository-style settings.

## Problem & Motivation

Existing code-generation benchmarks such as HumanEval and MBPP mostly test single-function synthesis from natural language, which poorly matches real software development where models must modify or complete existing code in context. Prior FIM evaluations are also small, mostly Python-only, and often rely on random span masking or prompt setups that unfairly favor certain model families. The paper addresses this gap by building a larger, multilingual, syntax-structured benchmark with stronger contamination controls and a more standardized evaluation protocol for comparing FIM-trained and non-FIM-trained code LLMs.

## Method

- **Benchmark construction**: SAFIM contains `17,720` examples from `8,590` Codeforces/GitHub code files written between `2022-04-01` and `2023-01-01`, chosen to reduce overlap with The Stack and GPT-3.5/GPT-4 training cutoffs.
- **Task splits**: the benchmark defines three AST-grounded completion types: algorithmic block completion (`8,781` examples), control-flow completion (`8,629`), and API function call completion (`310`).
- **Codeforces filtering**: candidate solutions are re-executed and kept only if they pass all tests within `50%` of the original time limit; very long solutions and near-duplicates with `CodeBLEU > 0.9` are removed.
- **GitHub filtering**: API-call examples are selected from repositories with more than `10` stars and must include enough comments or documentation to make the completion task solvable from context.
- **Prompting protocol**: the evaluation uses `5` prompt formats, `L2R`, `PSM`, `SPM`, `IPF`, and `1S`, to avoid prompt-specific bias across model families with different pretraining objectives.
- **Post-processing**: syntax-aware truncation uses AST validity checks to recover the intended completion. For block completion it iteratively removes trailing lines until the inserted span forms a valid block subtree; for expression-level tasks it incrementally grows the completion until it forms a valid expression.
- **Evaluation**: `98.25%` of examples use execution-based evaluation through unit tests, while API-call tasks use syntax-equivalence matching when execution is impractical because of external dependencies or side effects.
- **Inference setup**: GPT-3.5/GPT-4 are queried through the OpenAI API; open models are decoded with Hugging Face using `top-p = 0.95` and `temperature = 0.2`, and the paper reports single-sample `Pass@1`.

## Key Results

- SAFIM scale: `17,720` total examples across `4` languages, with average context length `3364B`; `98.25%` of the benchmark supports execution-based evaluation.
- Best overall model in the main comparison is DeepSeekCoder-33B with average `69.0 Pass@1`, including `60.8` on algorithmic blocks, `71.1` on control-flow, and `75.2` on API completion.
- FIM pretraining helps beyond infilling: StarCoder (`15.5B`) achieves `55.5` average Pass@1 and outperforms GPT-4's `53.3` average in the reported setup; CodeLLaMa-13B (`52.8`) also surpasses CodeLLaMa-34B (`49.7`).
- Syntax-aware truncation materially changes conclusions for open-ended models: CodeGen-16B rises from `0.0` to `25.9` Pass@1 on algorithmic block completion, while CodeLLaMa-13B rises from `16.4` to `41.4`.
- On control-flow completion, CodeLLaMa-13B improves from `27.8` to `57.2` with syntax-aware truncation, and DeepSeekCoder-33B reaches `71.1`.
- Additional contamination analysis on a newer post-`2023-04` dataset finds little degradation for CodeLLaMa and DeepSeekCoder, with deltas ranging from `+0.91` to `+5.29` on algorithmic block completion.

## Limitations

- The headline conclusions about pretraining paradigms come from cross-family comparisons rather than controlled ablations within a single architecture and training environment.
- The API function call split is much smaller than the execution-based splits (`310` examples), so conclusions about API knowledge are based on a narrower sample.
- Part of the benchmark timeline still overlaps with CodeLLaMa and DeepSeekCoder pretraining periods, so contamination risk is reduced but not eliminated.
- Reported results are sensitive to prompt choice and post-processing; the paper argues this is realistic, but it also means benchmark rankings depend on evaluation protocol design.

## Concepts Extracted

- [[large-language-model]]
- [[code-language-model]]
- [[fill-in-the-middle]]
- [[abstract-syntax-tree]]
- [[benchmark-dataset]]
- [[execution-based-evaluation]]
- [[prompt-engineering]]
- [[syntax-aware-truncation]]
- [[data-contamination]]
- [[data-curation]]
- [[data-deduplication]]

## Entities Extracted

- [[linyuan-gong]]
- [[sida-wang]]
- [[mostafa-elhoushi]]
- [[alvin-cheung]]
- [[safim]]
- [[codeforces]]
- [[github]]
- [[exec-eval]]
- [[humaneval]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
