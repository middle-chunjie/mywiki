---
type: entity
title: BabelTower
slug: babeltower
date: 2026-04-20
entity_type: tool
aliases: [Babel Tower]
tags: []
---

## Description

BabelTower is a prior unsupervised code-translation system for `C++ → CUDA` used as a baseline in [[tehranijamsaz-2024-coderosetta-2410-20527]]. The paper positions it as a strong but language-specific approach that depends on CUDA-oriented metrics and ranking.

## Key Contributions

- Serves as the main specialized baseline for `C++ → CUDA`, where CodeRosetta improves from `74.00` to `76.90` BLEU and from `77.12` to `78.84` CodeBLEU.
- Motivates CodeRosetta's metric-agnostic bidirectional design by illustrating the limits of task-specific evaluation machinery.

## Related Concepts

- [[code-translation]]
- [[back-translation]]
- [[sequence-to-sequence]]

## Sources

- [[tehranijamsaz-2024-coderosetta-2410-20527]]
