---
type: entity
title: Multilingual Grade School Math
slug: mgsm
date: 2026-04-20
entity_type: benchmark
aliases: [MGSM, Multilingual Grade School Math]
tags: []
---

## Description

Multilingual Grade School Math (MGSM) is a multilingual arithmetic reasoning benchmark derived from GSM8K. The paper includes it to test whether task-agnostic meta-prompting generalizes to multilingual math settings.

## Key Contributions

- Shows that meta-prompting is not uniformly stronger: the paper reports `84.8` with Python, close to standard prompting (`84.4`) and below multi-persona prompting (`85.7`).
- Helps expose that the scaffold's gains are concentrated more in decomposition- and tool-sensitive tasks than in every reasoning benchmark.

## Related Concepts

- [[arithmetic-reasoning]]
- [[large-language-model]]
- [[zero-shot-prompting]]

## Sources

- [[suzgun-2024-metaprompting-2401-12954]]
