---
type: concept
title: Program Synthesis
slug: program-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [program synthesis, 程序合成]
tags: [program-synthesis, search]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Program Synthesis** (程序合成) — the automatic construction of executable programs from specifications such as input-output examples, constraints, or formal tasks.

## Key Points

- [[fijalkow-2022-scaling]] studies program synthesis in a two-stage pipeline where neural predictions produce search guidance and symbolic procedures recover exact programs.
- The paper focuses on syntax-guided list-processing domains, using typed DSLs that are compiled into grammars before search begins.
- Its objective is not only correctness but search efficiency, measured by the expected number of candidate programs emitted before the target is found.
- The experiments show that better search back-ends substantially change practical synthesis performance even when the neural predictor is held fixed.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fijalkow-2022-scaling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fijalkow-2022-scaling]].
