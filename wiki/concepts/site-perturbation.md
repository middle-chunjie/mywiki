---
type: concept
title: Site Perturbation
slug: site-perturbation
date: 2026-04-20
updated: 2026-04-20
aliases: [perturbation content selection]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Site Perturbation** — the optimization problem of choosing what replacement or inserted token should be applied at each selected perturbation site.

## Key Points

- The paper assigns each site a one-hot vector `u_i` over the token vocabulary `Ω`, with `1^T u_i = 1`.
- Site perturbation is solved jointly with site selection, so the attack decides both location and content together.
- Earlier baselines optimized perturbation content but relied on random site choice, which limited attack quality.
- The paper reports that smoothing the perturbation variables `u` is especially helpful for improving the attack landscape.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[srikant-2021-generating-2103-11882]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[srikant-2021-generating-2103-11882]].
