---
type: entity
title: CodeContests
slug: codecontests
date: 2026-04-20
entity_type: benchmark
aliases: [Code Contests]
tags: [benchmark, code]
---

## Description

CodeContests is the competitive-programming benchmark used in [[gloeckle-2024-better-2404-19737]] to finetune and evaluate `7B` models pretrained with [[multi-token-prediction]].

## Key Contributions

- Provides the downstream finetuning task used to test whether MTP pretraining improves harder code-generation behavior beyond pretraining-only benchmarks.
- The paper evaluates `1000` samples per test problem across temperatures `0.5` to `0.9` and reports temperature-oracle `pass@k`.

## Related Concepts

- [[multi-token-prediction]]
- [[algorithmic-reasoning]]
- [[large-language-model]]

## Sources

- [[gloeckle-2024-better-2404-19737]]
