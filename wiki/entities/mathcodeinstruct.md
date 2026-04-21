---
type: entity
title: MathCodeInstruct
slug: mathcodeinstruct
date: 2026-04-20
entity_type: dataset
aliases: [MathCodeInstruct dataset]
tags: [dataset, math, instruction-tuning]
---

## Description

MathCodeInstruct is the `80k`-example instruction-tuning dataset introduced in [[unknown-nd-mathcoderseamless-2310-03731]], built from GSM8K, MATH, and interpolation-generated problems with LCE-style solutions.

## Key Contributions

- Starts from a `49k` seed set of GPT-4-generated solutions on GSM8K and MATH, then expands with synthetic interpolation problems to `80k` examples.
- Encodes each solution as interleaved natural-language, code, and execution blocks for MathCoder training.
- Improves CodeLlama-34B average benchmark score from `66.2%` to `70.2%` when interpolation data is included.

## Related Concepts

- [[synthetic-data-generation]]
- [[problem-interpolation]]
- [[lce-solution-format]]

## Sources

- [[unknown-nd-mathcoderseamless-2310-03731]]
