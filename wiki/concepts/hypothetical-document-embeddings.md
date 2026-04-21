---
type: concept
title: Hypothetical Document Embeddings
slug: hypothetical-document-embeddings
date: 2026-04-20
updated: 2026-04-20
aliases: [HyDE]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hypothetical Document Embeddings** — a zero-shot retrieval method that asks an LLM to generate a hypothetical answer document for a query and then retrieves using the representation of that generated text.

## Key Points

- This paper uses HyDE as the main LLM-based zero-shot baseline for comparison with LameR.
- The authors argue HyDE can still be bottlenecked by a weak self-supervised dense retriever even if the generated answer text is plausible.
- They also show that BM25 benefits less from vanilla HyDE prompting than Contriever does, motivating a more collection-aware prompting strategy.
- LameR can be viewed as replacing HyDE's query-only prompting with candidate-grounded prompting plus literal BM25 matching.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shen-2023-large-2304-14233]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shen-2023-large-2304-14233]].
