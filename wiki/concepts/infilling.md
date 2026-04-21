---
type: concept
title: Infilling
slug: infilling
date: 2026-04-20
updated: 2026-04-20
aliases: [text infilling, code infilling, 补全填充]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Infilling** (补全填充) — generation of a missing span inside an existing sequence while respecting the surrounding context on both sides.

## Key Points

- The paper studies free-form infilling rather than single-token cloze prediction, with most quantitative evaluation carried out on code.
- Infilling is harder than plain continuation because the model must both start compatibly with the prefix and stop in a way that connects to the suffix.
- The authors identify `<EOT>` prediction as a practical bottleneck: failing to emit it often causes overlong or suffix-incompatible completions.
- The work argues that realistic infilling quality must be measured by generated samples and task success, not only by intrinsic loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bavarian-2022-efficient-2207-14255]].
