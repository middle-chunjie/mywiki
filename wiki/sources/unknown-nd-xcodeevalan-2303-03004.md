---
type: source
subtype: paper
title: "xCODEEval: An Execution-Based Large-Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval"
slug: unknown-nd-xcodeevalan-2303-03004
date: 2026-04-20
language: en
tags: [benchmark, code-llm, multilingual, evaluation, retrieval]
processed: true

raw_file: raw/papers/unknown-nd-xcodeevalan-2303-03004/paper.pdf
raw_md: raw/papers/unknown-nd-xcodeevalan-2303-03004/paper.md
bibtex_file: raw/papers/unknown-nd-xcodeevalan-2303-03004/paper.bib
possibly_outdated: true

authors:
  - Mohammad Abdullah Matin Khan
  - M Saiful Bari
  - Xuan Long Do
  - Weishi Wang
  - Md Rizwan Parvez
  - Shafiq Joty
year: 2023
venue: arXiv preprint
venue_type: preprint
arxiv_id: 2303.03004
doi:
url: https://arxiv.org/abs/2303.03004
citation_key: unknownndxcodeevalan
paper_type: benchmark
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature. xCODEEval introduces a large multilingual benchmark for code-focused large language models that unifies code understanding, generation, translation, repair, and retrieval under execution-based evaluation rather than text-overlap scoring. Built from Codeforces submissions, the resource spans about `25M` examples, `16.5B` tokens, roughly `7.5K` problems, and seven tasks, while the accompanying [[exec-eval]] engine executes unit tests across many runtimes in a standardized secure environment. The paper also proposes a split-selection procedure that balances validation and test distributions over problem tags and outcomes. Baselines with `gpt-3.5-turbo-0301`, StarEncoder, StarCoderBase-3B, and CodeLlama show that the benchmark remains difficult, especially for low-resource languages, cross-lingual retrieval, and fully executable program synthesis.

## Problem & Motivation

Prior code-LLM benchmarks were fragmented across tasks, languages, and granularities, and many evaluated generated programs by lexical overlap rather than by whether the code actually worked. The authors aim to build a single document-level benchmark that supports multilingual code understanding and generation tasks with executable evaluation, enough training data for finetuning, and metadata rich enough to analyze language imbalance, task difficulty, and possible pretraining contamination. They also want a common infrastructure for secure large-scale execution so that functional correctness can be compared consistently across languages.

## Method

- **Data source and scale**: collect about `25M` code examples (`16.5B` tokens) from Codeforces, covering `7514` distinct competitive-programming problems and up to `11` executable languages in the main setup, with retrieval resources extended to `17` languages.
- **Held-out split creation**: reserve `N_h = 1354` problems, define `\gamma = |D_valid| / |D_test| = 1/5`, compute tag-wise ratios `\gamma_T`, and choose the split whose geometric mean over `{\gamma_T}` is closest to `\gamma`.
- **Balanced subset selection**: formulate validation/test down-selection as a circulation problem on a flow network `G = (V, E)` with lower and upper capacities `l(e)` and `c(e)` connecting source, problem, tag, and sink nodes.
- **Task suite**: define `7` tasks: Tag Classification, Code Compilation, Program Synthesis, Code Translation, Automatic Program Repair, Code-Code Retrieval, and NL-Code Retrieval.
- **Execution engine**: introduce [[exec-eval]], a Dockerized service with `44` compiler/interpreter runtimes and six outcomes: compilation error, runtime error, memory limit exceeded, time limit exceeded, wrong answer, and passed.
- **Evaluation metrics**: use `macro-F1` for [[tag-classification]], `accuracy` for [[code-compilation]], `pass@k` for generative tasks, and `Acc@k` for retrieval.
- **Program-synthesis probing**: sweep `20` temperatures over `0.0` to `2.0`, then use the best-performing fixed temperature near `0.32`; the main generation runs use `n = 20` samples for synthesis and `n = 10` for translation/APR.
- **Retrieval baseline**: finetune StarEncoder as a bi-encoder with `max_length = 1024`, effective batch size `48`, and `37` epochs for both code-code and NL-code retrieval.

## Key Results

- [[benchmark-dataset]] scale: `5,494,008` training examples for [[tag-classification]], `19,915,150` for [[code-compilation]], `5,538,841` for [[program-synthesis]] and [[code-translation]], `4,672,070` for [[program-repair]], plus retrieval train sizes of `45,270` (code-code) and `55,924` (NL-code).
- `gpt-3.5-turbo-0301` improves Tag Classification from `27.29` macro-F1 (code only) to `33.60` macro-F1 when a natural-language problem description is added.
- Code Compilation reaches `63.27` average accuracy, with stronger results on PHP (`76.47`) and Go (`70.28`) than on Java (`53.0`), Kotlin (`56.64`), and Rust (`54.26`).
- Program Synthesis remains hard: average `pass@5` is `30.48` at the fixed best temperature and `27.80` under the broader temperature-search setting; Automatic Program Repair is easier with average `pass@5 = 55.07`.
- Retrieval is substantially stronger for NL-Code than Code-Code: average `Acc@k` is `83.83` for NL-Code, versus `58.53` corpus-side and `56.41` query-side averages for Code-Code; appendix analysis reports `84.19` monolingual top-100 accuracy vs `56.93` cross-lingual.
- On smaller models, finetuned StarCoderBase-3B reaches `2.25` average `pass@5` on synthesis, outperforming CodeLlama-7B (`1.26`) but trailing CodeLlama-13B (`3.81`).

## Limitations

- The benchmark is built almost entirely from Codeforces, so domain diversity is narrow and centered on competitive-programming style reasoning.
- Language resources are heavily imbalanced; low-resource languages such as D, Ocaml, and Rust are consistently harder for both training and evaluation.
- Many samples are document-level, non-modular, and lack docstrings, which limits coverage of more structured software-engineering settings.
- Pretraining contamination remains a real concern; the paper can diagnose cutoff effects better than earlier benchmarks, but it cannot eliminate leakage by itself.
- The dataset was not fully human-audited; the authors report removing about `2M` samples with automated filters, yet residual sensitive content or insecure code may remain.
- The paper has minor internal count inconsistencies, such as references to `11` vs `17` languages in different sections, so some statistics require careful interpretation by task.

## Concepts Extracted

- [[benchmark-dataset]]
- [[multilingual-benchmark]]
- [[code-understanding]]
- [[code-generation]]
- [[program-synthesis]]
- [[code-translation]]
- [[program-repair]]
- [[code-search]]
- [[execution-based-evaluation]]
- [[code-compilation]]
- [[tag-classification]]
- [[knowledge-cutoff]]
- [[data-contamination]]

## Entities Extracted

- [[mohammad-abdullah-matin-khan]]
- [[m-saiful-bari]]
- [[xuan-long-do]]
- [[weishi-wang]]
- [[md-rizwan-parvez]]
- [[shafiq-joty]]
- [[islamic-university-of-technology]]
- [[nanyang-technological-university]]
- [[qatar-computing-research-institute]]
- [[bosch-research]]
- [[salesforce-research]]
- [[exec-eval]]
- [[codeforces]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
