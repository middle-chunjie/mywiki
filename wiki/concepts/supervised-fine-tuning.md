---
type: concept
title: Supervised Fine-Tuning
slug: supervised-fine-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [SFT, supervised finetuning, 监督微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Supervised Fine-Tuning** (监督微调) — instruction- or task-conditioned training on labeled prompt-response pairs used to adapt a pretrained language model toward assistant behavior.

## Key Points

- Phi-4 performs one SFT stage on about `8B` chat-formatted tokens with learning rate `1e-6`.
- The SFT mixture spans math, coding, reasoning, general conversation, model identity, safety, and multilingual supervision for `40` languages.
- In the post-training ablation, SFT alone already yields strong performance, but DPO stages add substantial gains on GPQA, MATH, ArenaHard, and PhiBench.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
