---
type: concept
title: Code Language Model
slug: code-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [Code LM, Code LLM, code language model, 代码语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Language Model** (代码语言模型) — a language model specialized for source code and programming tasks such as generation, explanation, debugging, and execution reasoning.

## Key Points

- [[ding-2024-semcoder-2406-01006]] argues that code LMs trained mostly on static source text miss deeper semantic information about runtime effects and state changes.
- SEMCODER starts from `DeepSeekCoder-6.7B` and adds semantic supervision so a `6.7B` code LM can compete with or beat larger open models on execution reasoning.
- The paper treats code generation, execution prediction, and debugging as related capabilities that benefit from a shared semantic training objective.
- The work positions code LMs as programming assistants that should understand what code does, not only how code looks syntactically.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-semcoder-2406-01006]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-semcoder-2406-01006]].
