---
type: concept
title: Prompt Formatting
slug: prompt-formatting
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt format, 提示格式]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Formatting** (提示格式) — the structural prompt tokens that specify how demonstrations and target inputs are arranged and what output form the model should produce.

## Key Points

- The paper finds that label anchors alone are not sufficient for context compression; formatting tokens must also be preserved.
- Hiddenanchor therefore concatenates hidden states of both label words and formatting tokens at each layer.
- Omitting formatting causes the model to prefer common vocabulary items over label words, indicating uncertainty about the output space.
- This result suggests that prompt structure and semantic anchors play complementary roles in in-context learning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-label]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-label]].
