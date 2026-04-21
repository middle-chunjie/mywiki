---
type: concept
title: Reasoning Language Model
slug: reasoning-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [reasoning LM, reasoning model]
tags: [llm, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reasoning Language Model** (推理语言模型) — a language model trained or prompted to spend test-time compute on explicit intermediate reasoning before producing a final answer.

## Key Points

- The paper studies reasoning models that improve as they generate longer chain-of-thought traces.
- Existing reasoning LMs often lack precise control over how much test-time compute they consume.
- L1 is built from a `1.5B` reasoning base model and adds prompt-conditioned control over reasoning length.
- The authors show that reasoning models can be trained to adapt their reasoning style to different token budgets.
- Short-budget reasoning can still outperform larger non-reasoning models when the training objective aligns length and accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
