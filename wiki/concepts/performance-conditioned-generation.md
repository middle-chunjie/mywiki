---
type: concept
title: Performance-Conditioned Generation
slug: performance-conditioned-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [goal-conditioned optimization]
tags: [llm, conditioning, code]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Performance-Conditioned Generation** — a generation scheme that conditions the model on an explicit target-performance tag so it can prefer higher-quality outputs during code optimization.

## Key Points

- The paper annotates each target program with a binned score in `{1, 2, ..., 10}` based on proximity to the fastest known solution for that task.
- During inference, the model is prompted with a maximal `10/10` tag to request the strongest optimization.
- This conditioning lets the model distinguish strong and weak human edits instead of learning from an undifferentiated mixture of targets.
- For CODELLAMA 13B, performance conditioning raises `Best@8` speedup from `3.43x` on HQ data to `5.65x`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shypula-2024-performanceimproving-2302-07867]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shypula-2024-performanceimproving-2302-07867]].
