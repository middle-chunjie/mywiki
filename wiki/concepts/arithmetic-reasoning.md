---
type: concept
title: Arithmetic Reasoning
slug: arithmetic-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [math reasoning, 算术推理]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Arithmetic Reasoning** (算术推理) — the ability of a model to interpret quantitative word problems and produce correct numerical answers through multi-step symbolic or verbal reasoning.

## Key Points

- The paper evaluates arithmetic reasoning on six datasets: GSM8K, SVAMP, MultiArith, AddSub, AQuA, and SingleEq.
- To support adapter fine-tuning, the authors build `Math10K` and use ChatGPT-generated chain-of-thought rationales after filtering incorrect answers.
- Arithmetic reasoning is where the paper's ID/OOD analysis is most nuanced: PEFT improves smaller LLMs strongly on simpler datasets, but difficult sets such as `GSM8K` and `AQuA` still trail GPT-3.5.
- `LLaMA-13B + LoRA` is the strongest math model in the paper, reaching `65.4%` average accuracy and surpassing GPT-3.5 on several simpler arithmetic benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hu-2023-llmadapters-2304-01933]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hu-2023-llmadapters-2304-01933]].
