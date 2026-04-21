---
type: concept
title: Salient Span Masking
slug: salient-span-masking
date: 2026-04-20
updated: 2026-04-20
aliases: [显著片段掩码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Salient Span Masking** (显著片段掩码) — a masking strategy that preferentially hides knowledge-bearing spans such as named entities or dates so a model must rely on external or global knowledge to recover them.

## Key Points

- REALM identifies named entities with a BERT-based tagger trained on [[conll-2003]] and dates with a regular expression.
- The method masks one salient span per sentence so the pre-training target more often requires world knowledge rather than local syntax alone.
- On NQ development, salient-span masking materially outperforms random token masking and random span masking.
- The paper argues that latent retrieval learning is especially sensitive to the consistency of this masking signal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guu-2020-realm-2002-08909]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guu-2020-realm-2002-08909]].
