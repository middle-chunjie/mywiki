---
type: concept
title: Visual Prompting
slug: visual-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [image prompting, prompt for vision models, 视觉提示]
tags: [computer-vision, prompting, in-context-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Visual Prompting** (视觉提示) — the practice of constructing a visual prompt (an image grid of input–output example pairs) to guide a large vision model toward a desired inference behavior without modifying model parameters.

## Key Points

- In the image inpainting framing (Bar et al., 2022), a prompt is a grid image where cells contain in-context image–label pairs alongside the query image.
- The grid accommodates up to 8 in-context examples; more examples generally improve performance.
- Prompt design determines how context is communicated to the model; the ordering of examples has minor variance but the identity of examples has major impact.
- Visual prompting is the vision analog of few-shot prompting in NLP, but the representation is spatial rather than token-based.
- Effective visual prompts require examples that are semantically and spatially close to the query (similar pose, background, appearance).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-nd-what]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-nd-what]].
