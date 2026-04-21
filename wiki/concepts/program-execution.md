---
type: concept
title: Program Execution
slug: program-execution
date: 2026-04-20
updated: 2026-04-20
aliases: [runtime behavior, 程序执行]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Program Execution** (程序执行) — the runtime realization of a program on concrete inputs, producing outputs and observable behaviors that reflect its functionality.

## Key Points

- The paper treats execution-derived input-output mappings as an explicit semantic view of code functionality.
- Program executions are summarized as multiple test cases rendered into natural language and concatenated to code during pre-training.
- The model never requires executions at inference time because dynamic information distillation transfers execution semantics into code-only embeddings.
- Small source-level differences that are hard to notice statically can still be separated when execution behavior changes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code-2309-09980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code-2309-09980]].
