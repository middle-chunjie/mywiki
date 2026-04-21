---
type: entity
title: Long Code Completion Benchmark
slug: lcc
date: 2026-04-20
entity_type: dataset
aliases: [LCC, Long Code Completion]
tags: []
---

## Description

LCC is the long-context code completion benchmark introduced in [[guo-2023-longcoder-2306-14893]]. It is built from GitHub code for Python, Java, and C# to stress file-level completion beyond the short contexts used by earlier public benchmarks.

## Key Contributions

- Provides `100k` train, `10k` dev, and `10k` test examples per language for long-context completion.
- Increases average test context length to roughly `1.8k–2.0k` tokens, about `5×` longer than prior public completion benchmarks.
- Serves as the main benchmark demonstrating LongCoder's advantage over dense and sparse baselines.

## Related Concepts

- [[code-completion]]
- [[long-context-modeling]]
- [[benchmark-evaluation]]

## Sources

- [[guo-2023-longcoder-2306-14893]]
