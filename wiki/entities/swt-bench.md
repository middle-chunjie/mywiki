---
type: entity
title: SWT-Bench
slug: swt-bench
date: 2026-04-20
entity_type: dataset
aliases: [SWT Bench]
tags: [benchmark, software-testing]
---

## Description

SWT-Bench is the benchmark introduced by this paper for issue-reproducing test generation in Python repositories. Each instance pairs a GitHub issue with a golden bug-fix patch and golden reference tests.

## Key Contributions

- Provides `1,983` executable benchmark instances and a `276`-instance Lite subset.
- Evaluates methods using both fail-to-pass success and patch-targeted change coverage.
- Recasts SWE-Bench-style repair data into a test-generation benchmark.

## Related Concepts

- [[benchmark-dataset]]
- [[test-case-generation]]
- [[issue-reproduction]]

## Sources

- [[m-ndler-2024-code-2406-12952]]
