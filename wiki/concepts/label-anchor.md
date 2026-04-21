---
type: concept
title: Label Anchor
slug: label-anchor
date: 2026-04-20
updated: 2026-04-20
aliases: [label words as anchors, 标签锚点]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Label Anchor** (标签锚点) — a demonstration label token whose representation aggregates task information early and later serves as a retrieval point for prediction.

## Key Points

- The paper hypothesizes that label words are the main carriers of demonstration semantics in in-context learning.
- Shallow-layer isolation of label words sharply harms output stability, suggesting that anchors collect upstream information.
- Deep-layer attention to anchors strongly correlates with the model's final class decision.
- Anchor re-weighting, anchor-only hidden-state compression, and anchor-distance error diagnosis all build directly on this abstraction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-label]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-label]].
