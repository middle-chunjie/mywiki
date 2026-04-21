---
type: concept
title: Skill Pruning
slug: skill-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [技能剪枝]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Skill Pruning** (技能剪枝) — the removal of low-value skills from a memory bank to keep retrieval efficient and preserve high-utility knowledge.

## Key Points

- D2Skill assigns each skill an eviction score based on utility and an exploration term derived from retrieval counts.
- When a bank exceeds capacity `N_max`, the lowest-scoring skills are removed until the bank fits the limit.
- Newly created skills are protected for `T_prot` steps so they are not deleted before receiving enough evaluation signal.
- The ablation without skill management drops ALFWORLD validation performance from `72.7` to `57.8`, indicating pruning is central rather than cosmetic.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tu-2026-dynamic-2603-28716]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tu-2026-dynamic-2603-28716]].
