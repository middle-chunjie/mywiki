---
type: concept
title: Self-Ask
slug: self-ask
date: 2026-04-20
updated: 2026-04-20
aliases: [self ask, self-ask prompting]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Ask** — a structured prompting method in which a language model explicitly proposes and answers follow-up questions before producing a final answer to a complex query.

## Key Points

- The prompt uses rigid scaffolds such as `Follow up:` and `Intermediate answer:` to separate decomposition from answering.
- Self-ask improves over standard chain-of-thought on Bamboogle, 2WikiMultiHopQA, and MuSiQue, with the largest absolute gain on Bamboogle.
- The structured format makes final answers easier to parse and reduces full-sentence answer-format failures relative to chain-of-thought.
- Because sub-questions are clearly delimited, self-ask can be augmented with a search engine without changing model weights or prompt format.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[press-2023-measuring-2210-03350]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[press-2023-measuring-2210-03350]].
