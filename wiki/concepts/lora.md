---
type: concept
title: LoRA
slug: lora
date: 2026-04-20
updated: 2026-04-20
aliases: [low-rank adaptation, Low-Rank Adaptation, 低秩适配]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**LoRA** (低秩适配) — a parameter-efficient fine-tuning method that freezes a pretrained weight matrix and learns a low-rank additive update through trainable factors.

## Key Points

- The paper uses LoRA as one of the PEFT settings compatible with MeZO, showing that forward-only optimization can target either full parameters or low-rank adapters.
- In the reported setup, LoRA uses `(r, \alpha) = (8, 16)` for both RoBERTa and OPT experiments.
- MeZO with LoRA is often competitive with full-parameter MeZO and can even outperform it on some tasks such as BoolQ in the OPT-13B table.
- Despite its storage efficiency relative to full fine-tuning, LoRA checkpoints are still larger than the seed-plus-projected-gradient reconstruction used by MeZO.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[malladi-2024-finetuning-2305-17333]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[malladi-2024-finetuning-2305-17333]].
