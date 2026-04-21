---
type: concept
title: Residual Quantization
slug: residual-quantization
date: 2026-04-20
updated: 2026-04-20
aliases: [RQ, residual vector quantization, 残差量化]
tags: [representation-learning, quantization]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Residual Quantization** (残差量化) — a multi-stage quantization scheme that encodes a vector by recursively quantizing the remaining residual after each codebook assignment.

## Key Points

- ToolWeaver uses an RQ-VAE-style design so each tool receives a sequence of discrete codes instead of one flat identifier.
- At each level, the model chooses the nearest centroid to the current residual and passes the remaining error to the next codebook.
- This hierarchical decomposition gives much larger representation capacity than a single flat codebook while keeping vocabulary size compact.
- The paper argues that early code levels capture broad collaborative semantics and later levels refine finer functional distinctions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2026-toolweaver-2601-21947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2026-toolweaver-2601-21947]].
