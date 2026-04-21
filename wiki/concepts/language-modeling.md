---
type: concept
title: Language Modeling
slug: language-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [LM, 语言建模]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Language Modeling** (语言建模) — the task of assigning probabilities to token sequences, typically by predicting each token from its preceding context.

## Key Points

- REVELA reframes retriever learning as a language-modeling problem over document chunks.
- The LM is not merely a teacher; it is co-trained with the retriever under the same next-token prediction objective.
- Cross-document context is injected into language modeling through an additional in-batch attention pathway.
- The paper reports that the co-trained LM largely preserves its original capabilities, suggesting the auxiliary retrieval pathway does not heavily damage LM behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cai-2026-revela-2506-16552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cai-2026-revela-2506-16552]].
