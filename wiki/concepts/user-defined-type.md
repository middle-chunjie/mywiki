---
type: concept
title: User-defined Type
slug: user-defined-type
date: 2026-04-20
updated: 2026-04-20
aliases: [custom type, 用户定义类型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**User-defined Type** (用户定义类型) — a class or named-tuple type created by developers rather than provided as a built-in type by the language.

## Key Points

- HiTYPER separates user-defined types from built-in and generic types because they require different inference support.
- The paper reports that most rare types in the evaluated datasets are in fact user-defined types.
- Import analysis collects available user-defined classes so invalid neural predictions can be corrected against a valid local candidate set.
- Supporting user-defined types is a major reason the framework improves over both prior static tools and neural baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
