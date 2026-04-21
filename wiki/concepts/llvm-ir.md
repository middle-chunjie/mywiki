---
type: concept
title: LLVM IR
slug: llvm-ir
date: 2026-04-20
updated: 2026-04-20
aliases: [LLVM Intermediate Representation, LLVM 中间表示]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**LLVM IR** (LLVM 中间表示) — the intermediate representation used by the LLVM compiler framework to encode program semantics independently of source language and target machine.

## Key Points

- XLIR uses LLVM-IR as the common representation for C, C++, Java, and decompiled binaries.
- Source code is translated into LLVM-IR with LLVM Clang, JLang, and Polyglot, while binaries are converted with RetDec.
- The paper uses LLVM bitcode as the embedding input because it is more efficient for machine processing than human-readable MIR.
- Debug information is stripped during compilation to avoid trivial lexical leakage from source symbols.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gui-2022-crosslanguage-2201-07420]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gui-2022-crosslanguage-2201-07420]].
