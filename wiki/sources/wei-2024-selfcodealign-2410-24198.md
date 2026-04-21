---
type: source
subtype: paper
title: "SelfCodeAlign: Self-Alignment for Code Generation"
slug: wei-2024-selfcodealign-2410-24198
date: 2026-04-20
language: en
tags: [code-llm, instruction-tuning, self-alignment, synthetic-data, code-generation]
processed: true

raw_file: raw/papers/wei-2024-selfcodealign-2410-24198/paper.pdf
raw_md: raw/papers/wei-2024-selfcodealign-2410-24198/paper.md
bibtex_file: raw/papers/wei-2024-selfcodealign-2410-24198/paper.bib
possibly_outdated: false

authors:
  - Yuxiang Wei
  - Federico Cassano
  - Jiawei Liu
  - Yifeng Ding
  - Naman Jain
  - Zachary Mueller
  - Harm de Vries
  - Leandro von Werra
  - Arjun Guha
  - Lingming Zhang
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2410.24198
doi:
url: http://arxiv.org/abs/2410.24198
citation_key: wei2024selfcodealign
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

SelfCodeAlign proposes a fully transparent pipeline for aligning code large language models using only the base model itself during data generation, response sampling, and validation. Starting from `250k` filtered Python functions with docstrings from The Stack V1, the method extracts coding concepts, generates diverse instructions with few-shot prompting, samples multiple `(response, tests)` pairs per instruction, and keeps only responses that pass sandbox execution. In the main CodeQwen1.5-7B experiment, the pipeline produces `74k` instruction-response pairs and yields a finetuned model that reaches `67.1` `pass@1` on HumanEval+, outperforming CodeLlama-70B-Instruct and the same backbone trained on OctoPack. The paper's main claim is that execution-filtered self-alignment can beat both natural-data tuning and proprietary-model distillation for code generation.

## Problem & Motivation

Open code LLMs are usually instruction-tuned either with costly human-written supervision or with outputs distilled from stronger proprietary models. The authors argue that both routes are limiting: human annotation is expensive, distillation can violate licensing constraints, and teacher-generated data can introduce distribution shift that is suboptimal for the student model.

The paper targets a more transparent and permissive alternative. Its premise is that a strong base code model already contains enough knowledge to bootstrap useful instruction-following data, provided the pipeline enforces diversity at the task-generation stage and correctness at the response-selection stage. SelfCodeAlign is designed to make that self-generated supervision reliable enough to improve downstream code generation and editing.

## Method

- **Seed curation**: start from The Stack V1 and keep `250k` Python functions from about `5M` candidates with docstrings after Pyright checking, benchmark removal, documentation filtering, and near-deduplication via MinHash/LSH with Jaccard threshold `0.5`.
- **Task schema**: build `21` few-shot exemplars of `(seed, property, instruction, response, tests)`, where properties encode coding concepts, programming language, task category, and difficulty.
- **Instruction generation**: use Self-OSS-Instruct with `8-shot` prompting for concept extraction and instruction generation; task category is sampled from function/class/program implementation and difficulty from `easy|medium|hard`.
- **Response generation**: use `1-shot` prompting to sample multiple `(response, tests)` outputs per instruction, explicitly asking the model to emit executable tests together with the response.
- **Diversity control**: when a test specification has the right format, include it in the instruction body with probability `0.5` to diversify the final instruction distribution.
- **Sampling temperature**: run the generation pipeline at temperature `0.7` to balance sample quality and diversity.
- **Execution filtering**: execute generated tests in a Docker sandbox with common Python libraries; discard failing responses and randomly keep one passing response per instruction for training.
- **Training setup**: finetune self-generated data with learning rate `1e-5`, warmup ratio `0.05`, linear scheduler, Adafactor, batch size `64`, and sequence length `1280`; when training on data from other teacher models, use learning rate `2e-5`.
- **Compute**: main generation and training run on `4 x A100` machines; CodeQwen1.5-7B with `37k` seeds yields about `16k` accepted samples in `7h 30min`, while the main paper-scale experiment reports a final dataset of `74k` instruction-response pairs.

## Key Results

- On HumanEval+, SelfCodeAlign-CQ-7B reaches `67.1` `pass@1`, beating CodeQwen1.5-7B-Base (`45.7`) by `+21.4`, CodeQwen1.5-7B-OctoPack (`50.6`) by `+16.5`, and CodeLlama-70B-Instruct (`65.2`) while being about `10x` smaller.
- On MBPP+, the same model scores `65.2`; on LiveCodeBench it scores `22.4`, `22.8`, and `23.4` for start dates `2023-09-01`, `2023-07-01`, and `2023-05-01`.
- On EvoEval, SelfCodeAlign-CQ-7B averages `43.6`, outperforming OctoPack (`42.2`) and most open baselines except CodeQwen1.5-7B-Chat (`48.0`).
- On EvalPerf, it reaches differential performance score `79.9`, `65.2` `pass@1`, and `54.0` win rate, ranking second among similarly sized evaluated models.
- On ClassEval it achieves `27.0` class-level and `52.6` method-level `pass@1`; on DS-1000 it reaches `39.1`; on CanItEdit it reaches `39.0`.
- Component analysis shows `passes only` response selection gives `65.2` on HumanEval+, versus `61.6` for random selection and `57.9` for failures only.
- Seed and concept ablations show `seed -> concepts -> instruction` with filtered functions reaches `65.2`, versus `59.8` for direct `seed -> instruction` and `64.0` with random snippets as seeds.
- Against teacher-based alternatives on matched `74k` datasets, SelfCodeAlign (`67.1`) beats Evol-Instruct (`59.1`), OSS-Instruct (`61.6`), and direct GPT-4o distillation (`65.9`).

## Limitations

- Data generation is bounded by an approximately `3000`-token window, so the training distribution is biased toward medium-length instruction-response pairs rather than long-context coding tasks.
- The pipeline discards negative samples after filtering; the paper notes these failures could be reused in reinforcement-learning-style training rather than thrown away.
- Correctness labels depend on model-generated unit tests, so false positives and false negatives remain possible when the generated tests are incomplete or incorrect.
- The evaluation suite is broad for code generation and editing but still stops short of harder settings such as complex program generation, repository-scale tasks, and agentic software engineering.

## Concepts Extracted

- [[self-alignment]]
- [[instruction-tuning]]
- [[synthetic-data-generation]]
- [[concept-extraction]]
- [[in-context-learning]]
- [[few-shot-learning]]
- [[self-validation]]
- [[execution-based-filtering]]
- [[test-case-generation]]
- [[code-generation]]

## Entities Extracted

- [[yuxiang-wei]]
- [[federico-cassano]]
- [[jiawei-liu]]
- [[yifeng-ding]]
- [[naman-jain]]
- [[zachary-mueller]]
- [[harm-de-vries]]
- [[leandro-von-werra]]
- [[arjun-guha]]
- [[lingming-zhang]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
