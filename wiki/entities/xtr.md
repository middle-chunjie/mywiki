---
type: entity
title: XTR
slug: xtr
date: 2026-04-20
entity_type: model
aliases: [ConteXtualized Token Retriever]
tags: []
---

## Description

XTR is the multi-vector retrieval model introduced in [[lee-nd-rethinking]]. It optimizes token retrieval directly and ranks candidates using retrieved token scores plus missing-similarity imputation.

## Key Contributions

- Removes the expensive gathering stage from ColBERT-style inference.
- Introduces an in-batch token retrieval objective aligned with first-stage candidate generation.
- Achieves state-of-the-art or near-state-of-the-art retrieval quality on BEIR, LoTTE, open-domain QA, and MIRACL with much cheaper scoring.

## Related Concepts

- [[multi-vector-retrieval]]
- [[token-retrieval]]
- [[missing-similarity-imputation]]

## Sources

- [[lee-nd-rethinking]]
