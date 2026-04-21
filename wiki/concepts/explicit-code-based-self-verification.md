---
type: concept
title: Explicit Code-Based Self-Verification
slug: explicit-code-based-self-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [CSV, code-based self-verification]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Explicit Code-Based Self-Verification** (显式代码自验证) — a prompting strategy in which a model solves a problem, writes code to verify the candidate answer, and revises the reasoning when verification fails.

## Key Points

- CSV augments a normal GPT4-Code solution with an explicit verification stage that returns one of `True`, `False`, or `Uncertain`.
- If verification is `False`, the model is prompted to amend its prior reasoning and generate a new solution rather than stopping at the first answer.
- The method is zero-shot and does not require an external verifier model, reward model, or task-specific few-shot exemplars.
- The paper shows that CSV alone raises GPT4-Code on MATH from `69.69%` to `73.54%`.
- Code-based CSV is more effective than natural-language self-verification, which slightly underperforms the unrestricted Basic Prompt baseline in the same paper.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-solving-2603-03507]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-solving-2603-03507]].
