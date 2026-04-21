---
type: concept
title: Candidate-Prompted Answer Generation
slug: candidate-prompted-answer-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [candidate prompting]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Candidate-Prompted Answer Generation** — prompting a language model with a query plus retrieved candidate passages so it generates answer text grounded in the target collection's domain and format.

## Key Points

- LameR retrieves `M = 10` candidate passages before calling the LLM and injects them into the prompt as possible answering passages.
- The candidates are useful even when many are wrong because they still expose in-collection genres, units, and lexical patterns.
- The paper frames the LLM as both a summarizer over correct candidates and a generator that can repair or rewrite them using its internal knowledge.
- The authors sample `N = 5` answers per query to increase lexical coverage and reduce vocabulary mismatch in the final retrieval stage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shen-2023-large-2304-14233]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shen-2023-large-2304-14233]].
