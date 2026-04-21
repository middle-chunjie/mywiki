---
type: concept
title: Fisher Information
slug: fisher-information
date: 2026-04-20
updated: 2026-04-20
aliases: [费舍尔信息]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Fisher Information** (费舍尔信息) — the amount of information an observation provides about an unknown parameter, often used to quantify how much uncertainty an item can reduce.

## Key Points

- For DAD item selection, the paper uses `I_i(\theta_j) = \gamma_i^2 p_{ij}(1 - p_{ij})` to score how informative an item is for a subject of ability `\theta_j`.
- Information is highest when the model is most uncertain about whether a subject will answer correctly, i.e. when `p_{ij}` is near `0.5`.
- The aggregate acquisition score is `Info(i) = \sum_j I_i(\theta_j)`, summing informativeness over subjects.
- Fisher-information-driven sampling improves low-budget ranking quality relative to random selection in the cold-start setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rodriguez-2021-evaluation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rodriguez-2021-evaluation]].
