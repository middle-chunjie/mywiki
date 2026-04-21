---
type: concept
title: Hard Label
slug: hard-label
date: 2026-04-20
updated: 2026-04-20
aliases: [binary label, 硬标签]
tags: [training, data-synthesis, classification]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hard Label** (硬标签) — a binary supervision signal that assigns each training instance a single definitive class (positive or negative) rather than a continuous probability or soft score.

## Key Points

- [[zhang-2025-lessons-2501-07301]] compares hard labels and soft labels for MC-estimation-based PRM training; hard labels treat a step as correct if *any* of 8 completions reaches the correct answer (`threshold = 0`).
- Before consensus filtering, the performance difference between hard and soft labels is small, masked by overall noise in MC estimation.
- After consensus filtering, hard labels substantially outperform soft labels on both BoN and PROCESSBENCH, because step correctness is a deterministic property and probabilistic soft labels introduce additional noise.
- Soft labels in MC estimation capture future-outcome probability rather than current-step correctness, misaligning with the deterministic nature of correctness evaluation.
- The threshold analysis shows that `threshold = 0` (positive if any completion succeeds) is the optimal hard-label setting across all threshold values from `1/8` to `7/8`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2025-lessons-2501-07301]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2025-lessons-2501-07301]].
