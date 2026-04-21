---
type: concept
title: Density Functional Theory
slug: density-functional-theory
date: 2026-04-20
updated: 2026-04-20
aliases: [密度泛函理论, DFT]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Density Functional Theory** (密度泛函理论) — a quantum-chemistry framework that estimates molecular or material properties from electron density rather than explicit many-body wavefunctions.

## Key Points

- The paper treats DFT as the dominant physics-based baseline for molecular property prediction and as the source of labels in QM9.
- It motivates MGCN partly because DFT calculations are computationally expensive, with the discussion citing near-hour-long prediction times for a single 20-atom molecule.
- The authors characterize DFT-based prediction as roughly `O(N^3)` in the number of particles, making large-scale screening difficult.
- MGCN is framed as a learned surrogate that preserves interaction structure while being much faster at inference.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lu-2019-molecular]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lu-2019-molecular]].
