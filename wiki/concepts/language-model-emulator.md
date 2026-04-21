---
type: concept
title: Language Model Emulator
slug: language-model-emulator
date: 2026-04-20
updated: 2026-04-20
aliases: [LMulator, language model emulator]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Language Model Emulator** (语言模型模拟执行器) — a language model used to simulate the effect of executing code when an interpreter cannot run a line directly.

## Key Points

- The paper coins the term "LMulator" for the LM component that predicts state updates for non-executable code lines.
- The LMulator receives the question, prior code, and current state history, then outputs the next program state rather than only a free-form answer.
- This mechanism is what lets CoC handle semantic predicates and pseudocode without abandoning the code-based reasoning format.
- The authors argue that LMulation is complementary to Python execution rather than a replacement for it.
- The same pattern is later used in the paper's robotics examples, where semantic judgments are simulated inline while APIs are executed directly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-chain-2312-04474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-chain-2312-04474]].
