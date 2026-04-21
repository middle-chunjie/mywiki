---
type: concept
title: Webpage Understanding
slug: webpage-understanding
date: 2026-04-20
updated: 2026-04-20
aliases: [web understanding, 网页理解]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Webpage Understanding** (网页理解) — the ability of a model to infer the content, structure, layout, and UI semantics of a webpage from its rendered representation.

## Key Points

- Web2Code treats webpage understanding as a distinct capability from code generation and evaluates it with the WUB benchmark.
- The paper argues the task requires multiple sub-skills simultaneously, including OCR, layout perception, spatial reasoning, and semantic interpretation of UI elements.
- WUB contains `1,198` webpage screenshots and `5,990` binary question-answer pairs generated with GPT-4V to probe fine-grained understanding.
- Training with webpage QA data improves not only WUB accuracy but also screenshot-to-HTML generation, suggesting shared representations between understanding and generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yun-2024-webcode-2406-20098]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yun-2024-webcode-2406-20098]].
