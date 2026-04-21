---
type: concept
title: Discrete Auto-Encoding
slug: discrete-autoencoding
date: 2026-04-20
updated: 2026-04-20
aliases: [discrete auto-encoding, 离散自编码]
tags: [representation-learning, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Discrete Auto-Encoding** (离散自编码) — an auto-encoding scheme in which inputs are compressed into discrete latent codes that must retain enough information for downstream reconstruction.

## Key Points

- [[sun-2023-tokenize-2304-04171]] uses discrete auto-encoding to learn semantic docids instead of hand-crafted document identifiers.
- The latent code is a multi-token docid `z`, and the reconstruction model retrieves the original document from prefix-matched candidates conditioned on that code.
- Training combines reconstruction, commitment, and retrieval losses so the discrete code supports both semantic fidelity and query-time retrieval.
- The paper uses a straight-through estimator because `argmax` over discrete docid tokens is non-differentiable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2023-tokenize-2304-04171]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2023-tokenize-2304-04171]].
