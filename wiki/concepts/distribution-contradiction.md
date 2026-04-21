---
type: concept
title: Distribution Contradiction
slug: distribution-contradiction
date: 2026-04-20
updated: 2026-04-20
aliases: [detriment term, 分布矛盾]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Distribution Contradiction** (分布矛盾) — the paper's name for the KL-divergence term that measures how strongly retrieved texts conflict with the LLM's knowledge in token prediction.

## Key Points

- It is formalized as `KL(p_R(r) || p(r | z*))`, where `z*` is the latent concept associated with retrieved texts.
- The paper interprets this term as token-level detriment, because noisy or wrong retrieval can pull prediction away from correct internal knowledge.
- Theorem 1 shows this term appears directly in the upper and lower bounds of `D = ||p_RAG - p_R||_1`.
- The theory predicts that stronger detriment makes it more important to reject RAG-selected tokens during collaborative decoding.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2025-theory-2406-00944]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2025-theory-2406-00944]].
