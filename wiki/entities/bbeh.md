---
type: entity
title: BBEH
slug: bbeh
date: 2026-04-20
entity_type: benchmark
aliases: [BIG-Bench Extra Hard, Big-Bench Extra Hard]
tags: [benchmark, llm]
---

## Description

BBEH is the `23`-task reasoning benchmark introduced in [[kazemi-2025-bigbench-2502-19187]] as a harder successor to [[bbh]] for evaluating broad LLM reasoning.

## Key Contributions

- Replaces every BBH task with a harder task in the same reasoning domain while preserving cross-domain coverage.
- Restores substantial evaluation headroom: the best general-purpose model reaches `9.8%` adjusted harmonic mean and the best reasoning-specialized model reaches `44.8%`.
- Includes a smaller `460`-example subset, BBEH Mini, for cheaper experimentation.

## Related Concepts

- [[benchmark-dataset]]
- [[benchmark-saturation]]
- [[harmonic-mean-evaluation]]

## Sources

- [[kazemi-2025-bigbench-2502-19187]]
