---
type: concept
title: Dual-Encoder Architecture
slug: dual-encoder-architecture
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 双编码器架构
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dual-Encoder Architecture** (双编码器架构) — a retrieval architecture that encodes queries and documents separately into comparable vector representations for efficient similarity search.

## Key Points

- LightRetriever starts from the standard dual-encoder setup but breaks the usual symmetry between query and document encoders.
- The document side remains a full LLM encoder so that document representations preserve deep contextual modeling.
- The query side is trained with a full encoder but served with cached token embeddings and averaging, reducing online cost dramatically.
- The paper shows that making both sides lightweight causes severe effectiveness drops, so asymmetry is central to the design.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2026-lightretriever-2505-12260]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2026-lightretriever-2505-12260]].
