---
type: entity
title: PIE Dataset
slug: pie-dataset
date: 2026-04-20
entity_type: dataset
aliases: [PIE, Performance-Improving Edits]
tags: []
---

## Description

PIE is the dataset introduced in the paper for studying performance-improving code edits. It pairs slower and faster accepted `C++` submissions with deterministic runtime labels and unit-test-based correctness checks.

## Key Contributions

- Contains `77,967` training pairs, `2,544` validation pairs, and `978` test pairs drawn from CodeNet problems.
- Keeps only optimization pairs with more than `10%` relative speedup.
- Supports prompting and fine-tuning experiments on program optimization with reproducible measurements.

## Related Concepts

- [[benchmark-dataset]]
- [[benchmark-reliability]]
- [[code-optimization]]

## Sources

- [[shypula-2024-performanceimproving-2302-07867]]
