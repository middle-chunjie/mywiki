---
type: concept
title: Code Editing
slug: code-editing
date: 2026-04-20
updated: 2026-04-20
aliases: [Code Editing, 代码编辑]
tags: [code, software-engineering, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Editing** (代码编辑) — the task of modifying existing code to implement a requested change while preserving unaffected behavior and context.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] introduces EDITINGRTC to evaluate code editing without manually labeled edit instructions.
- The forward model describes the difference between old and new code; the backward model applies that description to the old code.
- The paper uses exact match between reconstructed and original edited code as the main similarity function.
- The method targets realistic editing scenarios such as bug fixing, refactoring, and feature updates.
- Results on CodeReviewer show unsupervised RTC is more informative than lexical-overlap metrics for edit descriptions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
