---
type: concept
title: Relevance Scoring
slug: relevance-scoring
date: 2026-04-20
updated: 2026-04-20
aliases: [relevance estimation, relevance judgment, 相关性评分]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Relevance Scoring** (相关性评分) — assigning a numerical estimate of how well a candidate item matches a user, persona, or task context.

## Key Points

- GPT-4 assigns each question-persona pair a score `r in {1, ..., 10}` before click simulation.
- These scores are computed independently for each question to reduce bias from other items in the current pool.
- The paper validates this scoring stage with human comparison judgments and reports `77.3%` LLM alignment on agreed pairs.
- Relevance scores act as the latent utility signal that is later converted into CTR through the action model.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[senel-2024-generative-2406-05255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[senel-2024-generative-2406-05255]].
