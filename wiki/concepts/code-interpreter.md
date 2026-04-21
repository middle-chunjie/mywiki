---
type: concept
title: Code Interpreter
slug: code-interpreter
date: 2026-04-20
updated: 2026-04-20
aliases: [python interpreter, code interpreter]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Interpreter** (代码解释器) — a runtime system that executes generated programs or program fragments and returns their concrete computational effects.

## Key Points

- In this paper, Python is the concrete interpreter used to execute CoC lines when they are valid and runnable.
- The interpreter provides exact computation for arithmetic, symbolic updates, and API invocation, complementing LM-based semantic reasoning.
- The paper shows that interpreter use is essential for the strongest results on algorithmic tasks.
- Python-only execution is insufficient for many semantic tasks, motivating the hybrid design rather than a pure tool-use baseline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-chain-2312-04474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-chain-2312-04474]].
