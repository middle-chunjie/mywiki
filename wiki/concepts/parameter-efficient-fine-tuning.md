---
type: concept
title: Parameter-Efficient Fine-Tuning
slug: parameter-efficient-fine-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [PEFT, parameter efficient fine-tuning, 参数高效微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parameter-Efficient Fine-Tuning** (参数高效微调) — a family of adaptation methods that update a small set of additional or selected parameters instead of fully updating every weight in a pretrained model.

## Key Points

- QLoRA uses PEFT through LoRA adapters rather than full 16-bit end-to-end fine-tuning.
- The paper argues that PEFT is what makes 4-bit storage practical, because gradients only need to update adapter weights and not the frozen quantized backbone.
- In the memory analysis for LLaMA 7B on FLAN v2, LoRA parameters occupy only `26 MB`, while the 4-bit base model still dominates memory at `5,048 MB`.
- The study positions QLoRA as evidence that PEFT can recover much of the quality lost by low-bit storage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dettmers-2023-qlora-2305-14314]]
- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dettmers-2023-qlora-2305-14314]].
