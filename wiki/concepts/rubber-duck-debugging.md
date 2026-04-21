---
type: concept
title: Rubber Duck Debugging
slug: rubber-duck-debugging
date: 2026-04-20
updated: 2026-04-20
aliases: [rubber ducking, 橡皮鸭调试]
tags: [debugging, code]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Rubber Duck Debugging** (橡皮鸭调试) — a debugging practice where a programmer exposes mistakes by explaining code line by line, here adapted as a prompt format for language models.

## Key Points

- [[chen-2023-teaching-2304-05128]] explicitly frames SELF-DEBUGGING as an LLM version of rubber duck debugging.
- The paper uses natural-language explanation of generated code rather than explicit human-authored repair instructions.
- This explanation step is especially important on Spider, where no unit tests are available.
- The model compares its explanation against the task description to infer whether the code is actually correct.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-teaching-2304-05128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-teaching-2304-05128]].
