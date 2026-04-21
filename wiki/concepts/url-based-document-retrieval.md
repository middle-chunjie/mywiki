---
type: concept
title: URL-Based Document Retrieval
slug: url-based-document-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [LLM-URL, URL generation for retrieval, URL-as-document-identifier]
tags: [retrieval, url, llm, information-retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**URL-Based Document Retrieval** — a retrieval method in which a language model generates web URLs as document identifiers for a query; the documents are then fetched in real time via HTTP requests rather than retrieved from a pre-built offline index.

## Key Points

- URLs serve as semantically meaningful, human-interpretable document identifiers: for Wikipedia, the page title is embedded directly in the URL (e.g., `en.wikipedia.org/wiki/Jellyfish`), allowing an LLM to exploit this structure via its parametric knowledge.
- Compared to opaque autoregressive identifiers (atomic IDs, n-gram docids), URL identifiers carry lexical and topic signals that a pretrained LLM can leverage without task-specific fine-tuning.
- An LLM prompted with `"Which m Wikipedia URLs would have the answer? <question> https://en.wikipedia.org/wiki"` generates URLs in a zero-shot or few-shot setting; regex post-processing filters malformed or off-domain outputs.
- Real-time document fetching via HTTP GET makes the approach inherently time-sensitive: unlike offline corpora, pages that update in place are retrieved with current content, enabling handling of time-sensitive queries without retraining.
- The validity rate of generated URLs is >68% for up to `m=10` generations, but falls off with larger `m`; this limits the effective recall ceiling.
- Wikipedia URL generation achieves better Recall@1 than title-only generation, suggesting the structured URL format itself cues more precise generation.

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ziems-2023-large-2305-09612]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ziems-2023-large-2305-09612]].
