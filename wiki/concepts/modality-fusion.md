---
type: concept
title: Modality Fusion
slug: modality-fusion
date: 2026-04-20
updated: 2026-04-20
aliases: [模态融合, multimodal fusion]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Modality fusion** (模态融合) — the integration of representations from different modalities into a shared model space so one model can condition on signals that were not originally expressed in text.

## Key Points

- xRAG treats dense retriever embeddings as a retrieval modality analogous to image or audio features in multimodal language models.
- The paper uses a two-layer MLP projector `W` to map a document embedding into the LLM embedding space, then prepends that projected vector as one token.
- Paraphrase pretraining aligns projected retrieval features with document text before downstream instruction tuning.
- Context-aware instruction tuning and self-distillation teach the frozen LLM to use the fused retrieval feature rather than a textual passage.
- The design goal is practical plug-and-play fusion: only the projector is trained, while both the retriever and the LLM remain frozen.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-xrag-2405-13792]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-xrag-2405-13792]].
