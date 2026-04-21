---
type: concept
title: LLM Evaluator
slug: llm-evaluator
date: 2026-04-20
updated: 2026-04-20
aliases: [LLM-as-a-judge, LLM judge, 大语言模型评测器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**LLM Evaluator** (大语言模型评测器) — a language model used as a judge to compare or score candidate outputs, typically conditioned on an instruction and an evaluation prompt.

## Key Points

- The paper defines an LLM evaluator as a combination of a base LLM and a prompting strategy rather than as the base model alone.
- Evaluator quality varies sharply across models on LLMBar, with GPT-4 substantially stronger than ChatGPT, LLaMA-2-70B-Chat, and Falcon-180B-Chat on adversarial instruction-following judgments.
- Prompt design materially changes evaluator behavior: Rules, self-generated Metrics, and Reference prompts improve adversarial accuracy, while Swap improves positional agreement.
- High scores on prior meta-evaluation sets can hide weakness on objective instruction-following comparisons, so evaluator selection needs benchmark-specific validation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-evaluating-2310-07641]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-evaluating-2310-07641]].
