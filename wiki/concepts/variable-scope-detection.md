---
type: concept
title: Variable Scope Detection
slug: variable-scope-detection
date: 2026-04-20
aliases: [Scope Detection, variable-scope-prediction]
tags: [code-intelligence, program-analysis, evaluation]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Variable Scope Detection** (变量作用域检测) — a binary classification probe task in which a model must determine whether two variables in a program share the same lexical scope, used to evaluate whether a code representation model captures block-level structural hierarchy.

## Key Points

- Designed in HiT as a diagnostic task to validate that global hierarchy embeddings encode accurate block-structure information; not a standalone application task.
- Given representations `h_A` and `h_B` for two variable occurrences, the scope probability is computed as `p_scope = sigmoid(h_A W_s h_B)` with a learned bilinear parameter `W_s`.
- Dataset construction from Python800 and C++1400 (Project CodeNet): ~7M pairs for Python, ~65M for C++, balanced positive/negative.
- HiT outperforms vanilla Transformer by ~3.1% (Python) and ~12.4% (C++), and outperforms GREAT by ~0.9% (Python) and ~6.7% (C++), confirming that global hierarchy captures scoping information.
- The larger gap on C++ reflects longer, more deeply nested programs where sequence models struggle to track scope without explicit structural signals.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-implant-2303-07826]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-implant-2303-07826]].
