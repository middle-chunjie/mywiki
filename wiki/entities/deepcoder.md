---
type: entity
title: DeepCoder
slug: deepcoder
date: 2026-04-20
entity_type: tool
aliases: [DeepCoder]
tags: []
---

## Description

DeepCoder is the neural-guided program synthesis system introduced in [[balog-2017-deepcoder-1611-01989]]. It predicts DSL function usage from input-output examples and uses those predictions to accelerate symbolic search.

## Key Contributions

- Instantiates the Learning Inductive Program Synthesis framework with a compact DSL, synthetic data generation, and a feed-forward predictor.
- Integrates learned guidance with DFS, sort-and-add enumeration, [[sketch]], and [[lambda-squared]].
- Demonstrates large runtime speedups over unguided baselines on small program-synthesis benchmarks.

## Related Concepts

- [[inductive-program-synthesis]]
- [[neural-guided-search]]
- [[enumerative-search]]
- [[domain-specific-language]]

## Sources

- [[balog-2017-deepcoder-1611-01989]]
