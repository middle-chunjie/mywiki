---
type: concept
title: Document Truncation
slug: document-truncation
date: 2026-04-20
updated: 2026-04-20
aliases: [truncation, 文档截断]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document Truncation** (文档截断) — the fragmentation of a document into partial sequences such that some tokens are trained without the full local context that originally grounded them.

## Key Points

- The paper identifies concatenate-then-split pretraining as a major source of unnecessary truncation for documents that would otherwise fit inside the context window.
- Truncation removes grounding context and can force next-token prediction to rely on incomplete evidence.
- The authors connect truncation to faithfulness failures in summarization, weaker context following, and undefined-name errors in code generation.
- Best-fit Packing limits truncation to documents whose length genuinely exceeds the model context length.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-fewer-2404-10830]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-fewer-2404-10830]].
