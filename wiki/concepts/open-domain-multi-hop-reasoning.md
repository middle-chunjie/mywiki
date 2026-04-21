---
type: concept
title: Open-Domain Multi-Hop Reasoning
slug: open-domain-multi-hop-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [ODMR, open domain multi-hop reasoning, 开放域多跳推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Open-Domain Multi-Hop Reasoning** (开放域多跳推理) — answering a multi-hop question without a provided candidate corpus while making the intermediate reasoning steps explicit.

## Key Points

- The paper introduces ODMR as a stricter extension of open-domain QA, where questions require `2-4` reasoning hops rather than mostly single-hop factual lookup.
- SP-CoT constructs ODMR examples automatically by generating linked QA quadruplets and composing them into multi-hop question chains.
- The generated ODMR dataset includes six reasoning graph types and both entity-answer and binary-question variants.
- The authors evaluate ODMR on ComplexWebQuestions, HotpotQA, 2WikiMultiHopQA, and MuSiQue after removing the provided supporting contexts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfprompted-2310-13552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfprompted-2310-13552]].
