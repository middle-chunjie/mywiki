---
type: entity
title: ToolTree
slug: tooltree
date: 2026-04-20
entity_type: tool
aliases: [ToolTree framework]
tags: []
---

## Description

ToolTree is the training-free LLM agent planning framework proposed in [[yang-2026-tooltree-2603-12740]]. It combines MCTS, dual LLM-based evaluation, and bidirectional pruning for multi-tool orchestration.

## Key Contributions

- Adds `r_pre` and `r_post` signals to search-time tool planning.
- Improves both benchmark accuracy and accuracy-per-second under fixed rollout budgets.
- Scales from closed-set tool suites to large open-set API libraries.

## Related Concepts

- [[tool-planning]]
- [[monte-carlo-tree-search]]
- [[bidirectional-pruning]]

## Sources

- [[yang-2026-tooltree-2603-12740]]
