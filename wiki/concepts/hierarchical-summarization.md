---
type: concept
title: Hierarchical Summarization
slug: hierarchical-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: [recursive summarization]
tags: [summarization, prompting, long-context]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Summarization** — a multi-stage summarization strategy that first summarizes local chunks and then recursively merges those intermediate summaries until one document-level summary remains.

## Key Points

- BOOOOKSCORE studies hierarchical summarization as one of two core workflows for books that exceed the context window.
- The method uses separate prompts for chunk summarization, summary merging, and merging with prior context from preceding summaries.
- Each merge step must fit within the model budget `W - G_l`, where `G_l` is the allowed output length for hierarchy level `l`.
- In the paper's experiments, hierarchical summarization is usually more coherent than incremental updating, especially for smaller-context models.
- The trade-off is reduced detail, because early local summaries can discard information before later merge steps see the whole narrative.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-booookscorea-2310-00785]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-booookscorea-2310-00785]].
