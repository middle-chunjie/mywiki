---
type: concept
title: Conversational Query Rewriting
slug: conversational-query-rewriting
date: 2026-04-20
updated: 2026-04-20
aliases: [对话式查询改写, conversational query reformulation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Conversational Query Rewriting** (对话式查询改写) — rewriting a context-dependent user utterance into a natural-language query that is explicit enough to be understood outside the immediate dialogue context.

## Key Points

- [[mao-2022-convtrans]] uses rewriting as one component of data augmentation rather than as the final retrieval interface.
- The paper decomposes rewriting into keyword-to-natural-language conversion (NL-T5) and natural-language-to-conversational conversion (CNL-T5).
- CNL-T5 conditions only on a short context, either the central query or a supporting sentence from the clicked passage, instead of the entire dialogue history.
- CANARD supplies the supervision for CNL-T5, using oracle queries from previous turns plus the current turn as input and the conversational query as target.
- The authors argue that narrower local context makes rewriting easier than prior methods that must process all previous turns.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mao-2022-convtrans]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mao-2022-convtrans]].
