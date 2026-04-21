---
type: concept
title: Structured Chain-of-Thought Prompting
slug: structured-chain-of-thought-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [SCoT prompting, structured CoT prompting, structured chain-of-thought prompting, 结构化思维链提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Structured Chain-of-Thought Prompting** (结构化思维链提示) — a two-stage prompting method for code generation that first asks an LLM to generate a Structured Chain-of-Thought and then asks it to synthesize code conditioned on that structure.

## Key Points

- The method uses separate prompts for SCoT generation and code generation, with `3` demonstrations per prompt by default.
- Stage 1 samples structured plans, while stage 2 converts each plan into code and can revise noisy SCoTs before implementation.
- The approach improves both ChatGPT and Codex on HumanEval, MBPP, and MBCPP relative to zero-shot, few-shot, and standard CoT prompting.
- Ablation results attribute most of the improvement to explicit program structures, with smaller but consistent gains from including input-output specifications.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structured-2305-06599]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structured-2305-06599]].
