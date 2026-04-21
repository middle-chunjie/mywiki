---
type: concept
title: Coherence Evaluation
slug: coherence-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [summary coherence evaluation]
tags: [evaluation, summarization, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Coherence Evaluation** — assessment of whether a generated summary is logically connected, understandable, and free of major omissions, discontinuities, contradictions, or redundant structure.

## Key Points

- The paper operationalizes coherence with a fine-grained, reference-free annotation protocol rather than overlap with gold summaries.
- Annotators mark confusing spans and ask clarification questions, yielding an eight-type taxonomy of coherence errors.
- Omission errors are the dominant failure mode for book-length summaries, especially under incremental updating.
- BOOOOKSCORE converts sentence-level coherence judgments into a system-level score by measuring the proportion of error-free sentences.
- Coherence and coarse human preference are related but not identical: more coherent summaries can still lose on perceived detail.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-booookscorea-2310-00785]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-booookscorea-2310-00785]].
