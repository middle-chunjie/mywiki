---
type: concept
title: Ensemble Learning
slug: ensemble-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [model ensemble, stacked ensemble, 集成学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Ensemble Learning** (集成学习) — the practice of combining multiple models or prediction streams so that their complementary strengths improve final performance.

## Key Points

- The paper tests a stacking-based ensemble built from four strong preprocessing configurations rather than from different architectures.
- Ensemble inputs come from `P_1101`, `P_0101`, `P_0110`, and `P_0111`, which preserve complementary token views of the same code.
- The ensemble improves over single settings for all reported models, reaching `10.64` for CodeNN, `11.28` for Astattgru, `24.21` for Rencos, and `19.90` for NCS.
- This result suggests that preprocessing diversity can be exploited as an additional modeling signal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-evaluation-2107-07112]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-evaluation-2107-07112]].
