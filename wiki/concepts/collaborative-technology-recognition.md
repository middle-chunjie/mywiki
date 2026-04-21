---
type: concept
title: Collaborative Technology Recognition
slug: collaborative-technology-recognition
date: 2026-04-20
updated: 2026-04-20
aliases: [CTR, 协同技术识别]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Collaborative Technology Recognition** (协同技术识别) — a method for identifying technologies that tend to co-occur or evolve together in patent activity.

## Key Points

- CTR constructs a yearly patent-technology bipartite graph and converts it into a weighted technology collaboration network.
- Collaboration strength is measured by overlap `|intersection| / |union|` between the patent sets of two technologies.
- The model retains the top-`n` collaborative technologies for each target technology in each year.
- These collaboration-weighted neighbors enrich the external-factor embedding used by DTT.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2019-deep]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2019-deep]].
