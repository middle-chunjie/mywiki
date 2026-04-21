---
type: concept
title: Intermediate Language
slug: intermediate-language
date: 2026-04-20
updated: 2026-04-20
aliases: [intermediate representation language, 中间语言]
tags: [code-generation, compilation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Intermediate Language** (中间语言) - a representation layer between natural-language intent and a target language that is chosen to be easier for the model to generate and easier for a compiler or repair system to normalize.

## Key Points

- SPEAC separates the intermediate interface into a parent language `P` and a child language `C subset P`.
- The parent language should be natural for the LLM, while the child language should be easy to compile into the target VLPL.
- For the UCLID5 case study, `P = Python` and `C` is an object-oriented Python subset.
- Child-language methods correspond closely to UCLID5 constructs such as state declarations, initialization, and transition relations.
- Repair and hole-filling operate on the intermediate-language program before compilation to the target language.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mora-2024-synthetic-2406-03636]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mora-2024-synthetic-2406-03636]].
