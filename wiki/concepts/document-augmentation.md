---
type: concept
title: Document Augmentation
slug: document-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [文档增强]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document Augmentation** (文档增强) — a preprocessing strategy that rewrites a document and derives auxiliary supervision such as QA pairs so the model can internalize the document's knowledge more effectively.

## Key Points

- Parametric RAG augments each document by generating one rewrite and three QA pairs before training the document-specific adapter.
- The augmented data is used to train the document parameters with a standard next-token objective over document-question-answer concatenations.
- Ablations show that removing augmentation materially hurts downstream RAG accuracy compared with the full pipeline.
- The paper finds QA generation contributes more than rewriting alone, but the combination of both produces the strongest overall performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2025-parametric-2501-15915]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2025-parametric-2501-15915]].
