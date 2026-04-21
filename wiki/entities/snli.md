---
type: entity
title: SNLI
slug: snli
date: 2026-04-20
entity_type: dataset
aliases: [Stanford Natural Language Inference, Stanford NLI]
tags: []
---

## Description

SNLI is a natural language inference dataset used by supervised SimCSE as a source of entailment positives and contradiction hard negatives. It provides manually written premise-hypothesis pairs with richer semantic variation than paraphrase-style corpora.

## Key Contributions

- Supplies supervised positive pairs through entailment annotations.
- Supplies contradiction examples that SimCSE reuses as hard negatives.
- Contributes to the `86.2` STS-B dev result when combined with MNLI under the final supervised setup.

## Related Concepts

- [[natural-language-inference]]
- [[hard-negative-sampling]]
- [[sentence-embedding]]

## Sources

- [[gao-2022-simcse-2104-08821]]
