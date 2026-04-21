---
type: concept
title: Verifiable Reward
slug: verifiable-reward
date: 2026-04-20
updated: 2026-04-20
aliases: [RLVR, reward with replayable evidence]
tags: [reinforcement-learning, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Verifiable Reward** (可验证奖励) — a reward signal whose components can be reconstructed from externally checkable artifacts such as tool transcripts, schema checks, deterministic tests, and logged evidence.

## Key Points

- ASG-SI decomposes reward into tool validity, outcome verification, skill reuse, composition integrity, and memory discipline.
- Each component is intended to be reconstructible from the verifier's evidence bundle instead of opaque preference judgments.
- The paper motivates verifiable rewards as a guard against reward hacking and measurement artifacts in agent training.
- Reward shaping is staged so early optimization emphasizes structural validity before later phases emphasize correctness and efficiency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2025-audited-2512-23760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2025-audited-2512-23760]].
