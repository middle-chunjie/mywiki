---
type: concept
title: Zero-Shot Prompting
slug: zero-shot-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [zero-shot prompting, 零样本提示]
tags: [prompting, generation, zero-shot]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Prompting** (零样本提示) — conditioning a language model with task instructions or context alone, without providing labeled examples or task-specific finetuning.

## Key Points

- The paper chooses a simple prompt template instead of few-shot, chain-of-thought, or ReAct prompting because the generators are small and input length is expensive.
- Each prompt contains the selected article text plus a question prefix such as `What`, `How`, `Where`, `Is`, or `Why`.
- Generated outputs are accepted as candidate questions only if they end with a question mark, providing a lightweight format constraint.
- The work positions zero-shot prompting as a cheaper alternative to supervised question generators and expensive proprietary LLM APIs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
