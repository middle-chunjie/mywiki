---
type: entity
title: ReProver
slug: reprover
date: 2026-04-20
entity_type: tool
aliases: [Retrieval-Augmented Prover, retrieval augmented prover]
tags: []
---

## Description

ReProver is the retrieval-augmented theorem prover introduced in [[yang-nd-leandojo]]. It combines a premise retriever and a ByT5 tactic generator with search to prove Lean theorems against the mathlib library.

## Key Contributions

- Restricts retrieval to accessible premises and uses in-file hard negatives during retriever training.
- Improves Pass@1 over both a no-retrieval generator and a zero-shot GPT-4 baseline on LeanDojo Benchmark.
- Transfers to MiniF2F and ProofNet despite being trained on static LeanDojo data.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[dense-passage-retrieval]]
- [[best-first-search]]

## Sources

- [[yang-nd-leandojo]]
