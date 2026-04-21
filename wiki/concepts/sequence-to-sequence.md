---
type: concept
title: Sequence-to-Sequence Learning
slug: sequence-to-sequence
date: 2026-04-20
updated: 2026-04-20
aliases: [Seq2Seq, sequence-to-sequence]
tags: [architecture, generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sequence-to-Sequence Learning** (序列到序列学习) — a modeling framework that maps an input sequence to an output sequence, typically with an encoder producing representations and a decoder generating the target autoregressively.

## Key Points

- [[ahmad-2021-unified]] frames code summarization, code generation, and code translation as sequence-to-sequence problems over code and natural language.
- PLBART adopts a seq2seq [[transformer]] with a bidirectional encoder and autoregressive decoder, unlike encoder-only code models that need a randomly initialized decoder at fine-tuning time.
- The same pretrained backbone is reused across source-code-to-text, text-to-code, and code-to-code transfer.
- The paper argues that unified seq2seq pre-training is especially useful when labeled parallel data is scarce.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ahmad-2021-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ahmad-2021-unified]].
