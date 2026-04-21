---
type: concept
title: Session Data Generation
slug: session-data-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [Synthetic Session Generation, 会话数据生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Session Data Generation** (会话数据生成) — the automatic construction of multi-turn conversational sessions and their supervision signals to expand training data for conversational retrieval.

## Key Points

- ConvSDG generates full dialogue sessions from topic descriptions when relevance judgments are unavailable, rather than producing turns independently.
- The paper pairs generated sessions with pseudo labels by retrieving candidate passages and sampling `3` pseudo-positive documents from the top-`5` results for each turn.
- In the supervised setting, the framework also performs query-level augmentation by paraphrasing each original turn while reusing the original relevance judgments.
- The generated sessions are used to fine-tune a conversational dense retriever, showing that synthetic conversations can improve retrieval effectiveness on CAsT and TopiOCQA.
- The paper highlights a central tradeoff: more generated data generally helps, but noisy or distribution-shifting synthetic data may require filtering.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mo-2024-convsdg-2403-11335]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mo-2024-convsdg-2403-11335]].
