---
type: concept
title: Trustworthiness
slug: trustworthiness
date: 2026-04-20
updated: 2026-04-20
aliases: [reliability, 可信性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Trustworthiness** (可信性) — the extent to which a system produces answers that remain reliable under noisy evidence and are supported by the most credible available information.

## Key Points

- The paper frames trustworthiness as robustness to imperfect retrieval rather than only raw accuracy under clean settings.
- Astute RAG improves trustworthiness by comparing internal and external evidence instead of assuming retrieved context is always beneficial.
- Worst-case experiments on RGB show the method preserves performance much better than standard RAG when all retrieved documents are unhelpful.
- The method is designed to retain grounding benefits when retrieval is reliable while recovering from misleading or conflicting passages when it is not.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-astute-2410-07176]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-astute-2410-07176]].
