---
type: entity
title: TR-bench
slug: tr-bench
date: 2026-04-20
entity_type: benchmark
aliases: [TR benchmark, tool retrieval benchmark]
tags: []
---

## Description

TR-bench is the unified tool-retrieval benchmark introduced in [[xu-2024-enhancing-2406-17465]], combining ToolBench, T-Eval, and UltraTools for both in-domain and out-of-domain evaluation.

## Key Contributions

- Aggregates `195,937` training instructions from ToolBench and multiple held-out evaluation settings.
- Evaluates retrieval quality with `NDCG@1/3/5/10` under both in-domain and OOD conditions.

## Related Concepts

- [[tool-retrieval]]
- [[benchmark]]
- [[cross-domain-evaluation]]

## Sources

- [[xu-2024-enhancing-2406-17465]]
