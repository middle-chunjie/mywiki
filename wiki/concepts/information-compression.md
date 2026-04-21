---
type: concept
title: Information Compression
slug: information-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [compression view of learning, compression-based inference]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Information Compression** — a learning perspective that evaluates models or prompt configurations by how efficiently they encode the target outputs conditioned on the inputs.

## Key Points

- The paper interprets a good in-context organization as one that losslessly compresses the target label with short codelength.
- This view motivates ranking candidate prompts by Shannon-Huffman codelength derived from model probabilities.
- Under the paper's approximation, compression connects directly to entropy minimization over predicted label distributions.
- The authors also relate the framework to cross-entropy when labels are known and to mutual information as an information-theoretic upper bound.
- The compression framing provides a principled explanation for why some demonstration orders are better than others.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2023-selfadaptive]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2023-selfadaptive]].
