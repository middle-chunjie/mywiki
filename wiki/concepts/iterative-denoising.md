---
type: concept
title: Iterative Denoising
slug: iterative-denoising
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative denoising, 迭代去噪]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Iterative Denoising** (迭代去噪) — a generation procedure that repeatedly refines a partially corrupted or uncertain representation over multiple steps until a clean output is produced.

## Key Points

- [[xiao-2026-embedding-2602-11047]] treats embedding inversion as iterative denoising over masked token sequences rather than iterative re-embedding and correction.
- Each denoising step updates all token positions simultaneously, which lets later context revise earlier guesses.
- The paper studies an explicit remasking policy that reopens the lowest-confidence `5%` of positions for additional refinement.
- The attack reaches strong token recovery with only `8` forward passes, making iterative denoising a latency-efficient alternative to many-step correction loops.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-11047]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-11047]].
