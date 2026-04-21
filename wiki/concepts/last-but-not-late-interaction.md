---
type: concept
title: Last-But-Not-Late Interaction
slug: last-but-not-late-interaction
date: 2026-04-20
updated: 2026-04-20
aliases: [LBNL interaction]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Last-But-Not-Late Interaction** — a reranking design in which query and candidate documents interact inside a shared transformer context during encoding, while final ranking embeddings are extracted from special token positions near the sequence end.

## Key Points

- The paper introduces LBNL as an alternative to classical late interaction, arguing that interaction should happen before document representations are finalized.
- Query and candidate documents are concatenated into one long prompt so causal self-attention can model both query-document and document-document relationships.
- The model extracts contextual states at `<|doc_emb|>` and `<|query_emb|>` positions rather than matching all document tokens directly.
- This design keeps listwise cross-document comparison while still using cosine similarity over compact projected embeddings for scoring.
- On BEIR, the LBNL-based reranker reports `61.85` average nDCG@10, outperforming the compared `0.6B` late-interaction baseline jina-colbert-v2 at `54.49`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2025-jinarerankerv-2509-25085]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2025-jinarerankerv-2509-25085]].
