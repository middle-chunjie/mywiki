---
type: concept
title: Parametric Memory
slug: parametric-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [parametric knowledge, model-internal knowledge]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parametric Memory** (参数记忆) — factual or procedural knowledge stored in a model's learned parameters and accessible without retrieving external evidence at inference time.

## Key Points

- The paper treats vanilla prompted LMs as systems relying purely on parametric memory for factual QA.
- Parametric memory is shown to work much better for popular entities than for long-tail entities.
- Scaling model size improves parametric recall mainly in the popularity head rather than deep in the tail.
- The authors argue that parametric memory alone is insufficient for robust factual QA because it can miss rare or obsolete facts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mallen-2023-when-2212-10511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mallen-2023-when-2212-10511]].
