---
type: concept
title: Document Tokenization
slug: document-tokenization
date: 2026-04-20
updated: 2026-04-20
aliases: [docid tokenization, 文档标记化]
tags: [retrieval, indexing]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document Tokenization** (文档标记化) — the process of mapping a document to a short discrete identifier sequence that can be used as a retrieval target instead of the original text.

## Key Points

- [[sun-2023-tokenize-2304-04171]] treats document tokenization as a learned component rather than a fixed preprocessing heuristic based on titles or clustering.
- The paper models tokenization with `Q(z|d)`, where each docid position is selected from a learned timestep-specific codebook.
- Good tokenization must keep docids short while preserving enough document semantics for accurate retrieval and reconstruction.
- The paper shows that better tokenization is the main reason generative retrieval improves on unseen documents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2023-tokenize-2304-04171]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2023-tokenize-2304-04171]].
