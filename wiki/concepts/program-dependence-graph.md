---
type: concept
title: Program Dependence Graph
slug: program-dependence-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [PDG, program dependence graph, 程序依赖图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Program Dependence Graph** (程序依赖图) — a graph representation of a program whose nodes are instructions or statements and whose edges encode control and data dependencies.

## Key Points

- The paper uses PDG as a sound over-approximation of the more expensive program interpretation graph `IG`.
- PDG edges include read-after-write, write-after-read, write-after-write, and control dependencies.
- Because `Aut(PDG)` is a supergroup of `Aut(IG)`, equivariance to PDG automorphisms implies equivariance to the target interpretation-graph automorphisms.
- SymC encodes PDG structure through shortest-path distance tuples and node degree features injected into multi-head attention.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
