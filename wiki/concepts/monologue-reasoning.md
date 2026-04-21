---
type: concept
title: Monologue Reasoning
slug: monologue-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [self-explanation reasoning, monologue reasoning, 独白式推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Monologue Reasoning** (独白式推理) — a code-reasoning format where a model explains program functionality, constraints, and execution behavior to itself in natural language.

## Key Points

- [[ding-2024-semcoder-2406-01006]] uses monologue reasoning as the central supervision signal that links source code to runtime semantics.
- The format includes approximate semantics, abstract properties, and operational explanations rather than only concrete traces or final answers.
- Forward monologue follows execution order and verbalizes line effects, variable changes, and final output prediction.
- Backward monologue reasons from observed outputs to abstract prior-state constraints, which helps when reverse execution is not uniquely determined.
- The paper reports that monologue reasoning outperforms scratchpad, NeXT traces, and concise traces on CRUXEval and LCB-Exec.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-semcoder-2406-01006]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-semcoder-2406-01006]].
