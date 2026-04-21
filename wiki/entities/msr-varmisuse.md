---
type: entity
title: MSR-VarMisuse
slug: msr-varmisuse
date: 2026-04-20
entity_type: dataset
aliases: [MSR VarMisuse]
tags: [benchmark, c-sharp]
---

## Description

MSR-VarMisuse is the C# benchmark used in [[vasic-2019-neural-1904-01720]] to compare the repair-only pointer model against prior graph-based VARMISUSE results. It contains file-level examples with type-compatible repair candidates.

## Key Contributions

- Provides the cross-paper comparison setting against Allamanis et al. (2018).
- Tests the syntax-only pointer model on a benchmark where slots include both load and store contexts.
- Shows the paper's pointer baseline reaching `62.3%` on the seen test split versus `55.3%` for the earlier syntax-only graph result.

## Related Concepts

- [[variable-misuse-bug]]
- [[graph-neural-network]]
- [[program-repair]]

## Sources

- [[vasic-2019-neural-1904-01720]]
