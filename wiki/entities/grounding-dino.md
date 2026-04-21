---
type: entity
title: Grounding DINO
slug: grounding-dino
date: 2026-04-20
entity_type: tool
aliases: [Grounding DINO]
tags: []
---

## Description

Grounding DINO is the referring-expression object detector used by VPEval in [[cho-2023-visual-2305-15328]] to localize prompt-specified objects. Its detections are paired with depth estimates to support spatial and scale reasoning.

## Key Contributions

- Powers the `objDet` primitive used for object, count, spatial, and scale evaluation.
- Produces bounding boxes that become part of VPEval's visual explanations.

## Related Concepts

- [[visual-programming]]
- [[layout-control]]
- [[explainable-evaluation]]

## Sources

- [[cho-2023-visual-2305-15328]]
