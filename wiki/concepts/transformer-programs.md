---
type: concept
title: Transformer Programs
slug: transformer-programs
date: 2026-04-20
updated: 2026-04-20
aliases: [Transformer Program, Transformer Programs, Transformer 程序]
tags: [transformer, interpretability, program-induction]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Transformer Programs** (Transformer 程序) — a constrained class of Transformers whose discrete components can be deterministically converted into human-readable programs while remaining trainable with gradient-based optimization.

## Key Points

- [[friedman-2023-transformer-2306-01128]] defines Transformer Programs by constraining residual representations, read/write structure, attention predicates, and feed-forward modules to stay in an interpretable discrete subspace.
- After training, the model is discretized by maximum-likelihood structure selection and converted into Python code with predicate functions and helper routines such as `selectclosest`.
- The framework supports categorical attention, numerical attention for counting-like behavior, and lookup-table-style MLPs that can be enumerated exactly.
- The learned programs are functionally identical to the trained model, enabling ordinary software debugging and static analysis.
- Empirically, the method remains competitive with standard Transformers on several short algorithmic and NLP tasks while yielding much more inspectable computations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-transformer-2306-01128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-transformer-2306-01128]].
