---
type: concept
title: LCE Solution Format
slug: lce-solution-format
date: 2026-04-20
updated: 2026-04-20
aliases: [LCE, natural-language-code-execution]
tags: [llm, reasoning, tool-use]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**LCE Solution Format** — a reasoning format that interleaves natural-language explanation, executable code, and execution outputs within one solution trace.

## Key Points

- [[unknown-nd-mathcoderseamless-2310-03731]] represents each solution as `y_i = (L, C, E, ...)`, where language blocks motivate code blocks and later language consumes execution outputs.
- The format is marked with explicit block tokens such as `<|text|>`, `<|code|>`, `<|execution|>`, and `<|endofblock|>`.
- During training, the model learns to emit both reasoning language and code, while execution-result tokens are masked out of the loss.
- During inference, generated code is run in a persistent notebook and the returned outputs are appended to the context before decoding continues.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-mathcoderseamless-2310-03731]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-mathcoderseamless-2310-03731]].
