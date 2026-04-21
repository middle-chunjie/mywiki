---
type: concept
title: Code Sketch
slug: code-sketch
date: 2026-04-20
updated: 2026-04-20
aliases: [code sketch, 代码草图]
tags: [code-generation, editing]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Sketch** (代码草图) — a partially specified code pattern that preserves reusable structure while masking details that must be filled in for a new requirement.

## Key Points

- In [[li-2023-skcoder-2302-06144]], the sketch is extracted from retrieved similar code rather than generated from scratch.
- The sketcher keeps requirement-relevant tokens and replaces irrelevant ones with `<pad>` placeholders, then merges consecutive placeholders.
- The paper argues that the sketch conveys "how to write" while the natural-language description specifies "what to write."
- The editor treats the sketch as a soft template and adds requirement-specific details such as conditions, identifiers, or extra parameters.
- Compared with copy-based augmentation, explicit sketches reduce blind copying of irrelevant code from retrieved examples.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-skcoder-2302-06144]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-skcoder-2302-06144]].
