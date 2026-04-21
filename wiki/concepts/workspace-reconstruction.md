---
type: concept
title: Workspace Reconstruction
slug: workspace-reconstruction
date: 2026-04-20
updated: 2026-04-20
aliases: [markovian workspace reconstruction, state reconstruction]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Workspace Reconstruction** (工作空间重建) — an agent design pattern that rebuilds the working context at each step from a compact state summary instead of appending the full interaction history.

## Key Points

- IterResearch reconstructs each round's workspace from the question, the evolving report `M_t`, and the latest tool interaction only.
- The paper contrasts mono-contextual `O(t)` context growth with iterative `O(1)` workspace structure under reconstruction.
- Historical tool traces are discarded after synthesis, so only compressed findings survive into the next round.
- The method is intended to prevent both context suffocation and noise contamination in long-horizon web research.
- Report writing is not a post-processing step; it is part of the policy output and therefore central to the state transition.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2025-iterresearch-2511-07327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2025-iterresearch-2511-07327]].
