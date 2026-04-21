---
type: concept
title: Information-Intensive Training
slug: information-intensive-training
date: 2026-04-20
updated: 2026-04-20
aliases: [In2 training, information intensive training, 信息密集训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Information-Intensive Training** (信息密集训练) — a data-centric long-context training strategy that concentrates answer-critical evidence into randomly placed short spans so the model must learn to retrieve and combine information from arbitrary positions in a long prompt.

## Key Points

- The method constructs long-context QA examples from C4 realnewslike by prompting GPT-4-Turbo on `~128`-token segments and shuffling the required segments with distractors into `4K`-to-`32K` contexts.
- It combines two supervision modes: fine-grained awareness of one short segment and reasoning over at least two distant segments.
- The final mixture contains `1.1M` fine-grained long-context examples, `300K` reasoning examples, `150K` short-context QA examples, and `200K` OpenOrca instruction examples.
- Applied to Mistral-7B-Instruct-v0.2, this recipe produces FilM-7B and sharply improves robustness on long-context probing tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[an-2024-make-2404-16811]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[an-2024-make-2404-16811]].
