---
type: concept
title: Molecular Property Prediction
slug: molecular-property-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [分子性质预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Molecular Property Prediction** (分子性质预测) — the task of estimating chemical or physical properties of molecules from their structure, composition, and geometry.

## Key Points

- The paper studies molecular properties such as HOMO, LUMO, energy-related quantities, heat capacity, dipole moment, and polarizability.
- ASGN formulates the task as regression from a molecular graph embedding to one or more real-valued properties.
- The motivating bottleneck is that high-quality property labels often come from computationally expensive DFT calculations.
- The framework aims to improve prediction accuracy when only a small labeled subset of chemical space is available.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hao-2020-asgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hao-2020-asgn]].
