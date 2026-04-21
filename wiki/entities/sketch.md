---
type: entity
title: Sketch
slug: sketch
date: 2026-04-20
entity_type: tool
aliases: [SKETCH, Sketch]
tags: []
---

## Description

Sketch is the SMT-based program synthesis tool used in [[balog-2017-deepcoder-1611-01989]] as one of the back-end solvers. The paper adapts it by treating function and argument choices as holes constrained by input-output consistency.

## Key Contributions

- Serves as an SMT-based baseline and target for neural guidance in DeepCoder.
- Benefits from sort-and-add restriction of the active DSL function set during search.

## Related Concepts

- [[satisfiability-modulo-theories]]
- [[neural-guided-search]]
- [[inductive-program-synthesis]]

## Sources

- [[balog-2017-deepcoder-1611-01989]]
