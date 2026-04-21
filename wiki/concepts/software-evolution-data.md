---
type: concept
title: Software Evolution Data
slug: software-evolution-data
date: 2026-04-20
updated: 2026-04-20
aliases: [open software evolution, 软件演化数据]
tags: [software-engineering, data]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Software Evolution Data** (软件演化数据) — historical records of how software systems change over time, including code snapshots, pull requests, issues, commits, and review interactions.

## Key Points

- The paper treats software evolution as the primary supervision source for reinforcement learning on real-world issue solving.
- Each training seed combines issue text, full code context, and the oracle patch merged in a pull request.
- The raw corpus is built from GitHub events plus repository histories rather than from curated benchmark-only datasets.
- The authors aggregate `24M` PR instances and filter them to about `11M` unique cases before selecting high-quality RL seeds.
- Using software evolution data lets the method avoid proprietary teacher traces while still exposing the model to realistic developer workflows.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wei-2025-swerl-2502-18449]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wei-2025-swerl-2502-18449]].
