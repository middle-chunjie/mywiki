---
type: entity
title: HiTYPER
slug: hityper
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

HiTYPER is a Python type inference framework that combines static analysis with neural type recommendations under a validation loop based on typing and rejection rules.

## Key Contributions

- Introduces type dependency graphs as the central representation for hybrid inference.
- Uses backward rejection to validate learned type suggestions before accepting them.
- Improves overall, rare-type, and return-value inference over prior neural and static baselines.

## Related Concepts

- [[hybrid-type-inference]]
- [[type-dependency-graph]]
- [[type-rejection]]

## Sources

- [[peng-2022-static-2105-03595]]
