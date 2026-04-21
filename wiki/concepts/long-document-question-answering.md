---
type: concept
title: Long-Document Question Answering
slug: long-document-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [long-qa]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Document Question Answering** (长文档问答) — question answering over a single long source document, where the model must locate and synthesize evidence from book-length or similarly large contexts.

## Key Points

- HELMET uses NarrativeQA and InfiniteBench QA/MC to test long-document reasoning at lengths that substantially exceed most older QA benchmarks.
- The benchmark controls evaluation length by truncating documents from the end to fit target context windows such as `8K` through `128K`.
- For NarrativeQA, the paper replaces overlap metrics with GPT-4o-based reference evaluation to better reflect answer quality.
- The paper shows that long-document QA trends do not fully align with synthetic recall or with citation-heavy tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2024-helmet-2410-02694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2024-helmet-2410-02694]].
