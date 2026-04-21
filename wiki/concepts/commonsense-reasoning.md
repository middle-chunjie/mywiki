---
type: concept
title: Commonsense Reasoning
slug: commonsense-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [commonsense qa, 常识推理]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Commonsense Reasoning** (常识推理) — the capability to solve tasks that rely on everyday physical, social, and causal knowledge rather than only local lexical matching.

## Key Points

- The paper evaluates commonsense reasoning on eight benchmarks: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, and OBQA.
- The authors construct `Commonsense170K` from the training sets of all eight tasks using explicit answer-format templates.
- Because the fine-tuning data covers the target tasks directly, commonsense reasoning becomes the paper's strongest in-distribution success case for PEFT.
- `LLaMA-13B + Parallel Adapter` reaches `81.5%` average accuracy on the eight-task suite, outperforming ChatGPT's `77.0%` average in the reported setup.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hu-2023-llmadapters-2304-01933]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hu-2023-llmadapters-2304-01933]].
