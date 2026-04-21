---
type: concept
title: Visual Programming
slug: visual-programming
date: 2026-04-20
updated: 2026-04-20
aliases: [visual programming, 视觉程序设计]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Visual Programming** (视觉程序设计) — a paradigm in which a language model or controller composes explicit visual-processing modules into an executable program for generation, reasoning, or evaluation.

## Key Points

- The paper uses visual programming on both sides of the pipeline: `VPGen` for generation and `VPEval` for evaluation.
- In `VPGen`, the program executes object/count generation, layout generation, and image generation as separate modules.
- In `VPEval`, the program dispatches specialized modules for object presence, counting, spatial relations, scale, and text rendering instead of using a single opaque scorer.
- The modular design makes intermediate outputs inspectable, such as predicted layouts, detected boxes, OCR strings, and module-level error messages.
- The paper argues that this decomposition also improves extensibility because modules can be swapped or extended without redesigning the full system.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cho-2023-visual-2305-15328]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cho-2023-visual-2305-15328]].
