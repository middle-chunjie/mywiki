---
type: concept
title: Local Variable Table
slug: local-variable-table
date: 2026-04-20
updated: 2026-04-20
aliases: [LocalVariableTable, 局部变量表]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Local Variable Table** (局部变量表) — metadata in bytecode disassembly that maps local-variable slots to variable names, scopes, and types for a method frame.

## Key Points

- TranCS extracts the LocalVariableTable together with the instruction sequence during disassembly.
- Slot indices in instructions are resolved against this table to recover human-readable variable names.
- The recovered variable names are inserted into translation templates through placeholders such as `[pv]`.
- Access to local-variable metadata helps preserve semantics that would be lost in raw opcode-only descriptions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-code-2202-08029]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-code-2202-08029]].
