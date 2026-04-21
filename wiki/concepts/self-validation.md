---
type: concept
title: Self-Validation
slug: self-validation
date: 2026-04-20
updated: 2026-04-20
aliases: [自我验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Validation** (自我验证) — explicit checking of intermediate or final reasoning outputs by the model to test whether they are consistent with the problem and earlier steps.

## Key Points

- The paper groups self-validation with reflection and backtracking as a defining component of long-CoT reasoning structure.
- Structurally coherent demonstrations teach the student to insert checking behavior into multi-step reasoning traces.
- Merely preserving keywords associated with checking is not enough; the checks must align with the surrounding steps to improve accuracy.
- The paper's perturbation studies suggest that self-validation is most useful when embedded in a logically ordered reasoning process.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-llms-2502-07374]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-llms-2502-07374]].
