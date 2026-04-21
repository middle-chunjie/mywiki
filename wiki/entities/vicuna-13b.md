---
type: entity
title: Vicuna 13B
slug: vicuna-13b
date: 2026-04-20
entity_type: tool
aliases: [Vicuna 13B, Vicuna]
tags: []
---

## Description

Vicuna 13B is the language model used in [[cho-2023-visual-2305-15328]] to generate object counts and text-serialized layouts before image rendering. The paper fine-tunes it with LoRA to become layout-aware while preserving pretrained knowledge.

## Key Contributions

- Generates object/count plans and bounding-box layouts in VPGen.
- Outperforms `GPT-3.5-Turbo + 36` in-context examples when used as the layout generator.
- Enables handling unseen object names through transfer from pretrained LM knowledge.

## Related Concepts

- [[low-rank-adaptation]]
- [[layout-control]]
- [[text-to-image-generation]]

## Sources

- [[cho-2023-visual-2305-15328]]
