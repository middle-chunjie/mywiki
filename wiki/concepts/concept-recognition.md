---
type: concept
title: Concept Recognition
slug: concept-recognition
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge concept matching, 概念识别]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Concept Recognition** (概念识别) — the process of mapping spans or tokens in text to concepts in an external knowledge resource so downstream models can operate over symbolic knowledge as well as words.

## Key Points

- [[liu-2023-enhancing]] performs concept recognition by soft n-gram matching with lemmatization and stop-word filtering against ConceptNet.
- The method constructs a token-aligned concept sequence `c`, assigning unmatched tokens to `[PAD]` and aligning n-gram concepts to the first token of the phrase.
- Recognized concepts are used twice: to build document-level concept sequences and to encode short label names for hierarchical label learning.
- The paper identifies recognition errors as a main source of noisy knowledge and one of the key limitations of the overall pipeline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-enhancing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-enhancing]].
