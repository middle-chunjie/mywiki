---
type: concept
title: Patch Well-Formedness
slug: patch-well-formedness
date: 2026-04-20
updated: 2026-04-20
aliases: [patch applicability, edit applicability, 补丁可应用性]
tags: [code-editing, evaluation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Patch Well-Formedness** (补丁可应用性) — the property that a model-generated patch can be applied to the target codebase without structural or formatting errors.

## Key Points

- The paper measures well-formedness as `W`, the proportion of instances where a generated patch applies successfully.
- Well-formedness is necessary but insufficient for issue-reproducing tests, so `W` should exceed downstream success metrics.
- Replacing unified diff hunks with the paper's custom function-level diff format raises applicability substantially for zero-shot generation.
- Robust patch formats matter because test generation still requires executable code edits even when the task is narrower than full repair.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[m-ndler-2024-code-2406-12952]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[m-ndler-2024-code-2406-12952]].
