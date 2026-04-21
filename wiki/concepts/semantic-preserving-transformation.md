---
type: concept
title: Semantic-Preserving Transformation
slug: semantic-preserving-transformation
date: 2026-04-20
updated: 2026-04-20
aliases: [semantics-preserving transformation, 语义保持变换]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic-Preserving Transformation** (语义保持变换) — a source-to-source program rewrite that changes syntax or expression form while preserving the original program semantics.

## Key Points

- The paper uses transformation families over control structures, APIs, and declarations to generate equivalent programs with different lexical and syntactic forms.
- It applies `10` such transformations for C/C++ and `9` for Java, enabling augmentation across multiple downstream datasets.
- Multiple transformations can be composed into longer transformation sequences, creating progressively harder variants of the same program.
- The paper's augmentation analysis suggests declaration and API rewrites are more useful than control-structure rewrites, which can add noise for masked-language-model backbones.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2022-bridging-2112-02268]]
- [[li-2022-ropgen-2202-06043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2022-bridging-2112-02268]].
