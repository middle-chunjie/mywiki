---
type: concept
title: Intermediate Representation
slug: intermediate-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [IR, 中间表示]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Intermediate Representation** (中间表示) — a program representation between source code and machine code that preserves semantics while normalizing away language- and platform-specific surface forms.

## Key Points

- This paper treats IR as the bridge that makes cross-language binary-source matching feasible.
- XLIR parses both source code and binaries into IR before learning a shared embedding space.
- The authors argue that IR reduces the semantic gap because different languages and architectures can map to structurally similar forms.
- The approach relies on compiler and decompiler tooling to produce IRs from both modalities.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gui-2022-crosslanguage-2201-07420]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gui-2022-crosslanguage-2201-07420]].
