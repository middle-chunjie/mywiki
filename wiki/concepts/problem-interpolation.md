---
type: concept
title: Problem Interpolation
slug: problem-interpolation
date: 2026-04-20
updated: 2026-04-20
aliases: [interpolation prompting, problem interpolation prompting]
tags: [llm, math-reasoning, data-generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Problem Interpolation** — a prompting strategy that generates a new problem whose difficulty lies between an easier seed problem and a harder seed problem.

## Key Points

- [[unknown-nd-mathcoderseamless-2310-03731]] uses a GSM8K example and a MATH example as anchors, then asks GPT-4 to synthesize a new intermediate-difficulty problem.
- The generated interpolation problems expand MathCodeInstruct beyond direct teacher solutions on existing benchmarks.
- The paper reports that `83.2%` of interpolation problems are harder than GSM8K anchors and `95.6%` are easier than MATH anchors.
- Adding interpolation data expands training from `49k` to `80k` examples and improves CodeLlama-34B average benchmark accuracy from `66.2%` to `70.2%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-mathcoderseamless-2310-03731]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-mathcoderseamless-2310-03731]].
