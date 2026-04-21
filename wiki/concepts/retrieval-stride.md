---
type: concept
title: Retrieval Stride
slug: retrieval-stride
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval frequency, 检索步幅]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval stride** (检索步幅) — the number of generated tokens between consecutive retrieval operations in a retrieval-augmented language model.

## Key Points

- [[ram-2023-incontext-2302-00083]] formalizes retrieval stride as `s`, where retrieval occurs once every `s` tokens instead of at every decoding step.
- Smaller `s` gives fresher grounding because retrieved documents are conditioned on a prefix closer to the upcoming tokens.
- The paper finds that perplexity improves as retrieval becomes more frequent, making low `s` preferable when runtime allows it.
- The reported default is `s = 4`, chosen as a compromise between language modeling quality and retrieval / recomputation cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ram-2023-incontext-2302-00083]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ram-2023-incontext-2302-00083]].
