---
type: concept
title: Webpage-to-Code Generation
slug: webpage-to-code-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [screenshot-to-html, webpage to code generation, 网页到代码生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Webpage-to-Code Generation** (网页到代码生成) — the task of translating a rendered webpage image into source code, typically HTML and related markup, that recreates the original visual and structural layout.

## Key Points

- Web2Code formulates the problem as instruction-following generation from webpage screenshots to HTML code.
- The paper shows that text-level code similarity is insufficient because many distinct HTML implementations can render the same page.
- Its WCGB benchmark therefore evaluates generated code by rendering it back to an image and scoring visual fidelity instead of comparing raw code strings.
- The Web2Code data mixture improves screenshot-to-HTML generation strongly across LLaMA3-8B, CrystalChat-7B, CrystalCoder-7B, and Vicuna1.5-7B backbones.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yun-2024-webcode-2406-20098]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yun-2024-webcode-2406-20098]].
