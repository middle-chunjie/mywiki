---
type: concept
title: Example Selection
slug: example-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [in-context example selection, demonstration selection, 示例选择]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Example Selection** (示例选择) — the process of choosing a small subset of candidate demonstrations that will most effectively help a model solve a new input.

## Key Points

- This paper treats example selection for code-generation ICL as the core bottleneck because prompt quality can shift performance from near-random to near-SOTA.
- LAIL first narrows the search space with BM25 to `50` candidate demonstrations and then re-ranks them using model-derived probability feedback.
- Helpfulness is defined operationally by how much a candidate raises the normalized log-probability of the target ground-truth program, not by surface lexical overlap alone.
- The selected top examples are used directly as prompt demonstrations at inference time, replacing heuristic random or BM25-only retrieval.
- Results show that learned selection yields consistent Pass@k gains over random, SBERT, GraphCodeBERT, and AceCoder baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-large-2310-09748]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-large-2310-09748]].
