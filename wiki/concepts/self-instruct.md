---
type: concept
title: Self-Instruct
slug: self-instruct
date: 2026-04-20
updated: 2026-04-20
aliases: [Self-Instruct]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Instruct** — a bootstrapping approach in which a language model generates instruction data that is later reused for supervising another round of instruction tuning.

## Key Points

- [[luo-2023-wizardcoder-2306-08568]] positions [[code-alpaca]] as a code-domain seed set generated with self-instruct style methods.
- The paper does not stop at self-generated seed data; it further evolves those instructions with [[evol-instruct]] to increase complexity and diversity.
- This framing separates two stages: initial synthetic instruction creation and subsequent complexity amplification for better downstream supervision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[luo-2023-wizardcoder-2306-08568]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[luo-2023-wizardcoder-2306-08568]].
