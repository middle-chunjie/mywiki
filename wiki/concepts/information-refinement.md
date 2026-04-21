---
type: concept
title: Information Refinement
slug: information-refinement
date: 2026-04-20
updated: 2026-04-20
aliases: [information refiner]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Information Refinement** (信息提炼) — the process of transforming retrieved or input text into output that is more concise, accurate, and complete by combining external evidence with model-internal knowledge.

## Key Points

- The paper treats information refinement as the core role of an LLM inside retrieval-augmented generation rather than simple conditional generation over retrieved passages.
- InFO-RAG operationalizes the concept through three unsupervised tasks: direct extraction, correction/completion under corrupted evidence, and generation from only semantically related context.
- The definition explicitly allows the model to fix errors and fill gaps in retrieved text instead of copying it verbatim.
- The authors argue that this view helps overcome the mismatch between decoder-only prefix language modeling and downstream RAG behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-unsupervised-2402-18150]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-unsupervised-2402-18150]].
