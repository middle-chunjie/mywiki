---
type: concept
title: Code Execution
slug: code-execution
date: 2026-04-20
updated: 2026-04-20
aliases: [program execution, 代码执行]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Execution** (代码执行) — the practice of running a reference program to obtain ground-truth behavior, verify model predictions, or generate executable supervision signals.

## Key Points

- CodeI/O executes transformed reference functions to produce input-output pairs for training rather than relying only on static code text.
- The pipeline skips functions with randomness and filters out samples whose execution exceeds `5` seconds or violates object-complexity constraints.
- Execution is used twice: first to construct the training data, and later to verify predicted outputs or predicted inputs.
- This executable grounding is what makes the paper's synthetic reasoning data scalable while still partially verifiable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-codeio-2502-07316]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-codeio-2502-07316]].
