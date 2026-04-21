---
type: entity
title: gem5
slug: gem5
date: 2026-04-20
entity_type: tool
aliases: [gem5 Simulator]
tags: []
---

## Description

gem5 is the full-system CPU simulator used in the paper to obtain deterministic execution-time measurements for code optimization. The authors use its Verbatim Intel Skylake configuration to reduce measurement noise that appears on commodity hardware.

## Key Contributions

- Provides deterministic benchmarking for PIE instead of relying on noisy hardware runtimes.
- Supports the paper's `42.8` million runtime simulations used to relabel performance data.
- Enables portability of the benchmark design to other architectures such as ARM or RISC-V.

## Related Concepts

- [[benchmark-reliability]]
- [[benchmark-dataset]]

## Sources

- [[shypula-2024-performanceimproving-2302-07867]]
