---
type: concept
title: Long Document Retrieval
slug: long-document-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [document retrieval over long texts]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long Document Retrieval** (长文档检索) — the task of encoding and ranking documents whose length exceeds standard passage limits while preserving both local evidence and document-level semantics.

## Key Points

- Longtriever frames long-document retrieval as a harder setting than passage retrieval because naive truncation loses relevant evidence and full attention is too expensive.
- The paper uses hierarchical block decomposition so the encoder can scale to documents of up to `8 x 512` tokens.
- Effective long-document retrieval requires both intra-block semantic modeling and explicit cross-block interaction.
- The paper argues annotation scarcity is more severe for this setting, making retrieval-oriented pretraining especially important.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-longtriever]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-longtriever]].
