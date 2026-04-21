---
type: concept
title: Frozen backbone
slug: frozen-backbone
date: 2026-04-20
updated: 2026-04-20
aliases: [backbone freezing, 冻结骨干网络]
tags: [training, transfer-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Frozen backbone** (冻结骨干网络) — a pre-trained feature extractor whose parameters remain fixed while only attached modules are optimized for the downstream task.

## Key Points

- FRPT freezes the entire pre-trained ResNet backbone and learns only prompt and adaptation modules.
- The first backbone block still supplies low-level features to the prompt module, but its weights are not updated.
- The last convolutional features serve as semantic input to CAH, again without changing backbone parameters.
- The paper argues that freezing preserves general visual discrimination that limited fine-grained datasets would otherwise degrade.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-finegrained-2207-14465]].
