---
type: concept
title: Code Explanation Generation
slug: code-explanation-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [explanatory code document generation, 代码解释生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Explanation Generation** (代码解释生成) — the task of generating detailed natural-language explanations of source code that cover both overall functionality and implementation-level behavior.

## Key Points

- [[cui-2022-codeexp-2211-15395]] distinguishes code explanation from short [[code-summarization]] by requiring parameter, return, and control-flow details.
- The paper formalizes the task as generating an explanatory docstring from a function's signature and body.
- The authors build dedicated training, refined, and annotated datasets because standard code-comment corpora are too noisy for this task.
- Human evaluation in the paper emphasizes adequacy, coverage, coherence, and fluency rather than only n-gram overlap.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cui-2022-codeexp-2211-15395]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cui-2022-codeexp-2211-15395]].
