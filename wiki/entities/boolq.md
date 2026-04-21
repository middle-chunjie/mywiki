---
type: entity
title: BoolQ
slug: boolq
date: 2026-04-20
entity_type: dataset
aliases: [Boolq, Boolean Questions]
tags: []
---

## Description

BoolQ is the yes/no question answering dataset used in [[lin-2023-batchprompt-2309-00384]] to evaluate how batched prompting affects accuracy, token use, and LLM call count.

## Key Contributions

- Serves as one of the three main classification benchmarks for BatchPrompt, BPE, and SEAS.
- Provides a headline result where batch size `32` with BPE+SEAS reaches `90.9%` accuracy against a `90.6%` SinglePrompt baseline.

## Related Concepts

- [[batch-prompting]]
- [[few-shot-prompting]]
- [[token-efficiency]]

## Sources

- [[lin-2023-batchprompt-2309-00384]]
