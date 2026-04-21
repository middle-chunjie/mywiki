---
type: concept
title: Document Parameterization
slug: document-parameterization
date: 2026-04-20
updated: 2026-04-20
aliases: [document parameterisation, 文档参数化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document Parameterization** (文档参数化) — the process of converting a raw document into a compact parameter block that can be inserted into a neural model as an explicit representation of the document's knowledge.

## Key Points

- Parametric RAG maps each corpus document `d_i` to a parametric representation `p_i = f_phi(d_i)` during an offline preprocessing stage.
- In this paper, the parametric representation is a document-specific LoRA update over feed-forward network weights rather than an embedding vector or retrieved text chunk.
- Training uses augmented document rewrites and generated QA pairs so the parameter block captures both factual content and answer-oriented usage patterns.
- Retrieved documents are merged at inference by summing their low-rank updates, allowing multiple documents to contribute to one temporary model state.
- The paper reports that a document representation for LLaMA-3-8B with rank `r = 2` occupies about `4.72 MB`, which is a practical trade-off between fast inference and storage cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2025-parametric-2501-15915]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2025-parametric-2501-15915]].
