---
type: entity
title: LLMBar
slug: llmbar
date: 2026-04-20
entity_type: tool
aliases: [LLM Bar]
tags: []
---

## Description

LLMBar is the instruction-following meta-evaluation benchmark introduced in [[unknown-nd-evaluating-2310-07641]]. It contains `419` pairwise comparison instances with objective preference labels, including both natural and adversarially curated subsets.

## Key Contributions

- Provides a benchmark for measuring whether LLM evaluators prefer the output that more faithfully follows the instruction.
- Separates a `100`-instance Natural split from a `319`-instance Adversarial split to expose evaluator weaknesses hidden by easier benchmarks.
- Supports analysis of evaluator accuracy, positional agreement, and prompt-design effects across proprietary and open models.

## Related Concepts

- [[instruction-following]]
- [[llm-evaluator]]
- [[meta-evaluation-benchmark]]
- [[adversarial-filtering]]

## Sources

- [[unknown-nd-evaluating-2310-07641]]
