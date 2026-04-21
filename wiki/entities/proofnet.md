---
type: entity
title: ProofNet
slug: proofnet
date: 2026-04-20
entity_type: dataset
aliases: [ProofNet benchmark]
tags: []
---

## Description

ProofNet is an external evaluation dataset referenced in [[yang-nd-leandojo]] for undergraduate-level mathematics theorem proving. In the paper it serves as a harder test of whether ReProver can prove theorems outside the mathlib training distribution.

## Key Contributions

- Supplies a test-only benchmark on which the paper reports the first theorem proving result, `13.8%` Pass@1.
- Reveals that premise retrieval is necessary for some newly discovered proofs contributed back to the dataset.

## Related Concepts

- [[automated-theorem-proving]]
- [[premise-selection]]
- [[data-splitting]]

## Sources

- [[yang-nd-leandojo]]
