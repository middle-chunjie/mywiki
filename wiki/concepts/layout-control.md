---
type: concept
title: Layout Control
slug: layout-control
date: 2026-04-20
updated: 2026-04-20
aliases: [layout control, 布局控制]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Layout Control** (布局控制) — explicit control over object positions, counts, and relative sizes through an intermediate representation that constrains image generation.

## Key Points

- `VPGen` improves layout control by predicting object counts and bounding boxes before invoking the image renderer.
- The layout is represented textually so the LM can leverage pretrained world knowledge, including unseen object names.
- The strongest gains appear on the skills most tightly coupled to layout fidelity: counting, spatial relations, and relative scale.
- The fine-grained analysis shows better performance on all tested count, spatial, and scale splits than Stable Diffusion v1.4.
- The paper also isolates layout-generation errors from layout-to-image rendering errors, showing that control can still be lost in the final diffusion step.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cho-2023-visual-2305-15328]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cho-2023-visual-2305-15328]].
