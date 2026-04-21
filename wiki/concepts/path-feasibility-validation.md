---
type: concept
title: Path Feasibility Validation
slug: path-feasibility-validation
date: 2026-04-20
updated: 2026-04-20
aliases: [path feasibility checking, 路径可行性验证]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Path Feasibility Validation** (路径可行性验证) — the verification of whether a candidate control-flow path and its induced dataflow fact can occur under satisfiable execution constraints.

## Key Points

- LLMDFA gathers branch conditions along stitched source-to-sink paths and translates them into logical constraints.
- The paper synthesizes Python scripts with Z3 bindings, then repairs them using execution errors from failed runs.
- Validation reaches `81.58%` precision and `99.20%` recall on DBZ, and `100.00%` precision and `100.00%` recall on XSS.
- Errors often arise on branch conditions involving library functions, user-defined predicates, or mutable global variables.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-when-2402-10754]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-when-2402-10754]].
