---
type: concept
title: External Memory
slug: external-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [外部记忆]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**External Memory** (外部记忆) — a non-parametric memory store outside model weights that retains reusable information for later retrieval during inference or training.

## Key Points

- The paper frames the skill bank as an external memory complementing the policy's parametric knowledge.
- Unlike raw trajectory storage, D2Skill stores abstracted task and step skills meant to transfer across tasks.
- Memory quality is maintained through utility updates, retrieval, and pruning rather than static caching.
- The related-work discussion places D2Skill in the broader shift toward structured reusable memory for evolving agents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tu-2026-dynamic-2603-28716]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tu-2026-dynamic-2603-28716]].
