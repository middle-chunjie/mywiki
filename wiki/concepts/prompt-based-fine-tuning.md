---
type: concept
title: Prompt-based Fine-tuning
slug: prompt-based-fine-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt tuning for classification, 基于提示的微调]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Prompt-based fine-tuning** (基于提示的微调) — a supervised adaptation strategy that reformulates a downstream task as masked-token prediction by wrapping the input in a natural-language prompt and mapping labels to label words.

## Key Points

- [[unknown-nd-improving-2401-02993]] follows the LM-BFF-style few-shot setup and predicts label words at a `[MASK]` position instead of training a standard classifier head.
- The paper argues that prompt-based NLU benefits from retrieval augmentation, but naive concatenation of retrieved texts makes prompts too long and expensive.
- ReFusion is designed specifically to improve prompt-based fine-tuning by injecting retrieved sentence representations directly into model internals.
- The reported experiments cover both single-sentence and sentence-pair prompt templates under the same prompt-based learning setup.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-improving-2401-02993]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-improving-2401-02993]].
