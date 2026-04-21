---
type: concept
title: Domain-Specific Language
slug: domain-specific-language
date: 2026-04-20
updated: 2026-04-20
aliases: [DSL, domain-specific language, 领域特定语言]
tags: [representation, program-synthesis]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Domain-Specific Language** (领域特定语言) — a restricted programming language designed for a narrow problem family so that valid programs are expressive enough for the target tasks but structured enough to search efficiently.

## Key Points

- [[balog-2017-deepcoder-1611-01989]] uses a list-processing DSL inspired by SQL and LINQ, with straight-line compositions over integers and integer arrays.
- The DSL contains high-level first-order and higher-order operations such as `Sort`, `Map`, `Filter`, `ZipWith`, and `Scanl1`, making function presence easier to predict from behavior.
- Programs execute as sequences of function calls over fresh variables, and the final variable is the output.
- The paper stresses that DSL choice controls both search difficulty and ranking ambiguity, so restrictiveness and expressiveness must be balanced.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[balog-2017-deepcoder-1611-01989]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[balog-2017-deepcoder-1611-01989]].
