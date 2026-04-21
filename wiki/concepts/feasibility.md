---
type: concept
title: Feasibility
slug: feasibility
date: 2026-04-20
updated: 2026-04-20
aliases: [solvability, 可解性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Feasibility** (可解性) — the item parameter that caps the maximum probability that any system can answer an example correctly, reflecting whether the item is actually solvable under the benchmark setup.

## Key Points

- In DAD, feasibility is `\lambda_i`, and low values indicate that even strong systems are prevented from achieving high success rates.
- The paper interprets `1 - \lambda_i` as the prevalence of annotation or task-design problems that make an item effectively unsolvable.
- Feasibility helps distinguish hard-but-valid items from invalid ones whose poor performance should not be credited to model weakness.
- The appendix reports that the lowest-feasibility tail in SQuAD 2.0 contains items with especially high likelihood of benchmark defects.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rodriguez-2021-evaluation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rodriguez-2021-evaluation]].
