---
type: entity
title: Ghidra
slug: ghidra
date: 2026-04-20
entity_type: tool
aliases: [Ghidra disassembler]
tags: []
---

## Description

Ghidra is the disassembler used in [[pei-2024-exploiting-2308-03312]] to lift binaries into P-Code for dependency computation in the binary-analysis pipeline.

## Key Contributions

- Supplies the binary lifting needed to build control and data dependencies for SymC.
- Supports the binary-analysis experiments on similarity, signature, and memory-region tasks.

## Related Concepts

- [[program-dependence-graph]]
- [[function-signature-prediction]]
- [[memory-region-prediction]]

## Sources

- [[pei-2024-exploiting-2308-03312]]
