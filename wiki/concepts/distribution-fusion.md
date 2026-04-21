---
type: concept
title: Distribution Fusion
slug: distribution-fusion
date: 2026-04-20
updated: 2026-04-20
aliases: [distribution mixture, 分布融合]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Distribution Fusion** (分布融合) — a formulation in which RAG next-token prediction is represented as the combination of the LLM's internal knowledge distribution and the distribution implied by retrieved texts.

## Key Points

- The paper models `p(x_i | R, x_{1:i-1})` as a latent-variable integral over concepts from the LLM and retrieved texts.
- The fusion weight is controlled by `v(z) = log ( p(R, x_{1:i-1} | z) / p(R, x_{1:i-1} | z*) )`, which determines whether the model leans toward retrieved knowledge or internal knowledge.
- Rewriting `v(z)` exposes benefit and detriment as the core forces governing the fusion.
- The formulation is the theoretical basis for later bounds on `D = ||p_RAG - p_R||_1` and for the Tok-RAG decoding rule.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2025-theory-2406-00944]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2025-theory-2406-00944]].
