---
type: concept
title: Prompt Sensitivity
slug: prompt-sensitivity
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt robustness, 提示敏感性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Sensitivity** (提示敏感性) — the extent to which a model's embedding behavior or downstream performance changes when the prompt wording changes.

## Key Points

- The paper evaluates nine prompt variants derived from PromptEOL-style instructions to test embedding stability.
- Hidden-state embeddings show higher variance across prompts, especially on STS12-14.
- Routing-weight embeddings maintain narrower performance ranges and higher cross-prompt correlation, indicating better robustness.
- MoEE benefits from prompting, but the paper argues that router-derived information is inherently less brittle than hidden-state-only embeddings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-your-2410-10814]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-your-2410-10814]].
