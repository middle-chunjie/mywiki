---
type: concept
title: Instruction Tuning
slug: instruction-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [instruction tuning, 指令微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Instruction Tuning** (指令微调) — a supervised training stage that adapts a pretrained model to follow natural-language instructions by optimizing it on instruction-response pairs.

## Key Points

- The paper performs In2 training in the instruction-tuning paradigm: the long context plus question are treated as the instruction, and only answer tokens contribute to the loss.
- FilM-7B is obtained by further fine-tuning Mistral-7B-Instruct-v0.2 for one epoch with global batch size `128`, `~14K` steps, cosine decay, and max learning rate `1e-6`.
- To avoid losing short-context behavior, the training mixture keeps `~9%` short-context QA data and adds `200K` OpenOrca examples.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[an-2024-make-2404-16811]]
- [[luo-2023-wizardcoder-2306-08568]]
- [[li-2024-infibench-2404-07940]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[an-2024-make-2404-16811]].
