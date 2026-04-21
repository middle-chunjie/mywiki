---
type: concept
title: Weighted Self-Consistency
slug: weighted-self-consistency
date: 2026-04-20
updated: 2026-04-20
aliases: [weighted self-consistency, 加权自一致性]
tags: [reasoning, decoding]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Weighted Self-Consistency** (加权自一致性) — a decoding strategy that aggregates multiple reasoning rollouts by weighting them with verifier scores instead of counting each rollout equally.

## Key Points

- The paper combines weighted self-consistency with a PRM trained from OmegaPRM annotations to improve final-answer selection on math reasoning tasks.
- Candidate solutions are not treated uniformly: each rollout receives a score from the product of its step-level PRM predictions before answer aggregation.
- This makes search sensitive to intermediate reasoning quality rather than only to how often an answer string appears among sampled rollouts.
- The paper reports consistent gains over unweighted majority voting and over PRMs trained on alternative supervision datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[luo-2024-improve-2406-06592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[luo-2024-improve-2406-06592]].
