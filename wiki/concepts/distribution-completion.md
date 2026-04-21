---
type: concept
title: Distribution Completion
slug: distribution-completion
date: 2026-04-20
updated: 2026-04-20
aliases: [benefit term, 分布补全]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Distribution Completion** (分布补全) — the paper's name for the KL-divergence term that measures how much retrieved texts add useful out-of-distribution knowledge beyond what the LLM already knows.

## Key Points

- It is formalized as `KL(p_R(r) || p(r | z))`, where `z` is a latent concept sampled from the LLM knowledge space.
- The paper interprets this term as the token-level benefit of retrieval in RAG.
- Larger distribution completion means the retrieved evidence covers knowledge that the LLM's internal distribution lacks.
- Tok-RAG does not estimate this term directly, but approximates its effect through representation similarity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2025-theory-2406-00944]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2025-theory-2406-00944]].
