---
type: concept
title: Parameter Sharing
slug: parameter-sharing
date: 2026-04-20
updated: 2026-04-20
aliases: [shared parameters, 参数共享]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parameter Sharing** (参数共享) — the design choice of reusing some neural model parameters across languages, tasks, or directions instead of allocating separate modules for each.

## Key Points

- The survey presents complete sharing and minimal sharing as two endpoints of the MNMT design space.
- Sharing can be controlled separately for embeddings, encoders, decoders, attention, and target-language-specific components.
- The right sharing pattern depends on language relatedness and the trade-off between model compactness and translation accuracy.
- The paper also discusses dynamic sharing, where language embeddings or routing networks determine how strongly parameters are shared.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dabre-2021-survey]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dabre-2021-survey]].
