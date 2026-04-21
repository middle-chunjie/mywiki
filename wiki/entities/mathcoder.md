---
type: entity
title: MathCoder
slug: mathcoder
date: 2026-04-20
entity_type: tool
aliases: [MathCoder-L, MathCoder-CL]
tags: [llm, math, reasoning]
---

## Description

MathCoder is the family of Llama-2- and CodeLlama-based models introduced in [[unknown-nd-mathcoderseamless-2310-03731]] to solve math problems with interleaved natural-language reasoning, code, and execution feedback.

## Key Contributions

- Achieves up to `83.9%` on GSM8K and `45.2%` on MATH among open-source models reported in the paper.
- Uses explicit block tokens and notebook-based code execution so the model can continue reasoning after observing runtime outputs.
- Includes both MathCoder-L and MathCoder-CL variants across `7B`, `13B`, `34B`, and `70B` scales.

## Related Concepts

- [[mathematical-reasoning]]
- [[code-execution]]
- [[lce-solution-format]]

## Sources

- [[unknown-nd-mathcoderseamless-2310-03731]]
