---
type: concept
title: Very Low-Resource Programming Language
slug: very-low-resource-programming-language
date: 2026-04-20
updated: 2026-04-20
aliases: [VLPL, very low-resource programming language, 极低资源编程语言]
tags: [code-generation, formal-methods]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Very Low-Resource Programming Language** (极低资源编程语言) - a programming or formal language with so little publicly available code and supervision data that pretrained models cannot reliably generate valid programs in it.

## Key Points

- The paper uses VLPL to describe targets such as internal DSLs, legacy tool-chain languages, and formal verification languages.
- UCLID5 is treated as a VLPL because examples are available only in the hundreds rather than thousands or millions.
- Direct zero-shot or few-shot generation in VLPLs often fails syntactically because the model lacks stable target-language priors.
- The paper argues that retrieval, constrained decoding, and fine-tuning each cover only part of the VLPL problem.
- SPEAC addresses VLPLs by moving generation into a higher-resource parent language and repairing back toward the target.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mora-2024-synthetic-2406-03636]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mora-2024-synthetic-2406-03636]].
