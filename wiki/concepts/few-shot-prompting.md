---
type: concept
title: Few-Shot Prompting
slug: few-shot-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [in-context demonstration prompting, 少样本提示]
tags: [llm, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Few-Shot Prompting** (少样本提示) — conditioning a model with a small number of demonstrations in-context so that subsequent predictions follow the desired format and task behavior.

## Key Points

- [[kadavath-2022-language-2207-05221]] uses few-shot examples to improve both answer formatting and calibration quality across several tasks.
- `20-shot` self-evaluation with comparison samples performs best among the paper's `P(True)` prompting setups.
- `10-shot` prompts are used for TriviaQA, Lambada, Arithmetic, and GSM8k when collecting samples for `P(IK)` training.
- The paper argues that few-shot prompting is especially important for calibration, not only for raw discrimination.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kadavath-2022-language-2207-05221]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kadavath-2022-language-2207-05221]].
