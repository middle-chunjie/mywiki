---
type: concept
title: Software Issue Resolution
slug: software-issue-resolution
date: 2026-04-20
updated: 2026-04-20
aliases: [issue solving, 软件问题求解]
tags: [software-engineering, bug-fixing]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Software Issue Resolution** (软件问题求解) — the end-to-end process of understanding a reported software issue, localizing relevant code, proposing edits, and validating that the fix resolves the problem.

## Key Points

- The paper frames SWE-bench style issue solving as the central downstream task for RL training.
- Each RL example asks the model to reason over an issue description and repository context, then emit concrete code edits.
- The learning objective targets realistic issue resolution rather than competitive-programming problem solving.
- Agentless Mini decomposes issue resolution into localization, repair, test generation, regression testing, and reranking.
- The authors report that RL on issue resolution alone still improves several out-of-domain reasoning tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wei-2025-swerl-2502-18449]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wei-2025-swerl-2502-18449]].
