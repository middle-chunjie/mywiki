---
type: concept
title: Text-to-Python Generation
slug: text-to-python-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [text-to-python, 文本到Python生成]
tags: [code-generation, python]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Text-to-Python Generation** (文本到Python生成) — the task of generating Python programs from natural-language problem descriptions, often under partial unit-test supervision.

## Key Points

- [[chen-2023-teaching-2304-05128]] evaluates this setting on [[mbpp]], where each problem has `3` tests and only the first is exposed in the prompt.
- The model must still judge whether a program is truly correct even after it passes the visible test.
- SELF-DEBUGGING improves Codex from `61.4` to `70.8` on MBPP and GPT-4 from `72.8` to `80.6`.
- The paper reports that many successful fixes address output mismatches and argument-order or argument-type mistakes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-teaching-2304-05128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-teaching-2304-05128]].
