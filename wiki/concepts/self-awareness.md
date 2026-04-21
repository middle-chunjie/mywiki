---
type: concept
title: Self-Awareness
slug: self-awareness
date: 2026-04-20
updated: 2026-04-20
aliases: [self-aware, self awareness, 自我认知]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Awareness** (自我认知) — the ability of a language model to judge whether it already knows the answer to a given question.

## Key Points

- UAR uses self-awareness to avoid unnecessary retrieval for non-time-sensitive questions that the model already knows.
- The paper builds model-specific IDK datasets by sampling `10` answers per TriviaQA question and labeling a question as known only when all responses are correct.
- Among UAR's four criteria, self-awareness is the hardest one and becomes the main performance bottleneck.
- The paper shows that replacing the lightweight head with a whole-LLM classifier yields only modest gains relative to the extra cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-unified-2406-12534]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-unified-2406-12534]].
