---
type: entity
title: APPS Benchmark
slug: apps-benchmark
date: 2026-04-20
entity_type: tool
aliases: [APPS]
tags: []
---

## Description

APPS Benchmark is the competitive-programming benchmark used in [[dainese-2024-generating-2405-15383]] to test GIF-MCTS as a general-purpose code synthesis method with unit-test feedback.

## Key Contributions

- Supplies `1000` competition-level Python problems for evaluating strict accuracy under a fixed call budget.
- Provides public unit tests that the paper uses as execution-based reward signals during search.

## Related Concepts

- [[code-synthesis]]
- [[program-repair]]
- [[chain-of-thought]]

## Sources

- [[dainese-2024-generating-2405-15383]]
