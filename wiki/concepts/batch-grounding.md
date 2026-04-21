---
type: concept
title: Batch Grounding
slug: batch-grounding
date: 2026-04-20
updated: 2026-04-20
aliases: [batched grounding]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Batch Grounding** — a grounding strategy that processes retrieved documents in small sequential batches, stopping once enough evidence is found to support or revise the current answer.

## Key Points

- GenGround uses batch grounding to avoid feeding long, noisy document lists into the LLM in a single pass during answer revision.
- The method checks batches `(1, b)`, `(b+1, 2b)`, and so on; the implementation sets `b = 3` while retrieving top-`10` documents per question.
- If a batch yields usable evidence, grounding terminates early; otherwise the model emits `Empty` and continues to the next batch.
- The ablation shows batch grounding materially improves HotpotQA, raising F1 from `47.27` to `52.26` and Acc from `45.03` to `47.27`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2024-generatethenground]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2024-generatethenground]].
