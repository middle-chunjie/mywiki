---
type: entity
title: QQP
slug: qqp
date: 2026-04-20
entity_type: dataset
aliases: [Quora Question Pairs]
tags: []
---

## Description

QQP is the duplicate question identification dataset used in [[lin-2023-batchprompt-2309-00384]] to test whether batched LLM prompting can stay accurate while reducing inference cost.

## Key Contributions

- Acts as the duplicate-detection benchmark in the paper's main evaluation suite.
- Shows a representative efficiency gain where batch size `32` with BPE+SEAS improves from `87.2%` to `88.4%` while using only `18.6%` of SinglePrompt's input tokens.

## Related Concepts

- [[batch-prompting]]
- [[in-context-learning]]
- [[token-efficiency]]

## Sources

- [[lin-2023-batchprompt-2309-00384]]
