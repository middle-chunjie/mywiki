---
type: entity
title: PopQA
slug: popqa
date: 2026-04-20
entity_type: dataset
aliases: [PopQA]
tags: []
---

## Description

PopQA is the long-tail open-domain QA benchmark evaluated in [[ma-nd-query-2305-14283]]. The paper uses a `13k`/`714` train-test split to study whether query rewriting helps when retrieval is useful only for some questions.

## Key Contributions

- Shows that both LLM and trainable rewriting outperform retrieve-then-read on long-tail QA.
- Reveals a limit of the trainable rewriter, which scores `45.72` EM / `49.51` F1 and remains slightly behind the frozen LLM rewriter.

## Related Concepts

- [[open-domain-question-answering]]
- [[query-rewriting]]
- [[retrieval-augmented-generation]]

## Sources

- [[ma-nd-query-2305-14283]]
