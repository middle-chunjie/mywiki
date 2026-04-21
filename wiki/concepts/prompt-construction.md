---
type: concept
title: Prompt Construction
slug: prompt-construction
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt design, 提示构造]
tags: [prompting, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Construction** (提示构造) — deliberate design of prompt structure and content to elicit or measure specific behaviors from a model.

## Key Points

- [[liu-nd-uncovering]] uses two complete functions plus one incomplete human-relevant signature as the minimal prompt structure that still exposes strong bias.
- Appendix results show elicited bias increases with more functions, more human-relevant context, and shorter distance between the human-relevant function and completion target.
- The paper demonstrates that even human-irrelevant seed functions can trigger biased completions if the final function signature is human-relevant.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-nd-uncovering]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-nd-uncovering]].
