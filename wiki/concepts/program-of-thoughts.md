---
type: concept
title: Program of Thoughts
slug: program-of-thoughts
date: 2026-04-20
updated: 2026-04-20
aliases: [PoT, program of thoughts]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Program of Thoughts** (程序化思维) — a prompting strategy that asks a language model to generate executable programs as intermediate reasoning traces, typically delegating computation to an external interpreter.

## Key Points

- This paper treats Program of Thoughts as a strong prior baseline for code-based reasoning.
- Program of Thoughts captures the benefit of executable code for numerical and symbolic tasks but assumes the intermediate program can actually run.
- Chain of Code is presented as a generalization that keeps executable code when possible while relaxing the requirement of full executability.
- The comparison helps motivate why semantic subroutines such as sarcasm or fruit judgments need LM-side simulation rather than pure Python execution.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-chain-2312-04474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-chain-2312-04474]].
