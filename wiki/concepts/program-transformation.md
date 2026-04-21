---
type: concept
title: Program Transformation
slug: program-transformation
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic-preserving transformation, 程序变换]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Program Transformation** (程序变换) — the rewriting of a program into an alternative form that preserves its intended semantics while altering surface structure.

## Key Points

- CODE-MVP treats transformed programs (`PT`) as one of the core views used during pretraining.
- The method builds positive pairs such as `PL` vs `PT` and `NL-PL` vs `NL-PT` to align functionally equivalent variants.
- Program transformations are motivated as a way to encode functional information that is not fully visible from raw tokens or ASTs alone.
- In the ablation study, removing the `PT` view lowers average retrieval MRR from `54.4` to `52.5`, showing that transformed variants contribute measurable signal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2022-codemvp-2205-02029]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2022-codemvp-2205-02029]].
