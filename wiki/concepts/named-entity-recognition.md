---
type: concept
title: Named Entity Recognition
slug: named-entity-recognition
date: 2026-04-20
updated: 2026-04-20
aliases: [Named Entity Recognition, NER, 命名实体识别]
tags: [nlp, sequence-labeling, information-extraction]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Named Entity Recognition** (命名实体识别) — the sequence-labeling task of assigning entity-type tags such as person, organization, or location to spans in text.

## Key Points

- [[friedman-2023-transformer-2306-01128]] uses English CoNLL-2003 NER as one of its main real-world evaluations for Transformer Programs.
- In this setup, each word is labeled under an IOB2 scheme with one of four entity types (`PER`, `ORG`, `LOC`, `MISC`) or as non-entity.
- The paper uses only categorical attention heads and one categorical MLP per layer for this task, with `300`-dimensional GloVe initialization.
- The resulting Transformer Program reaches `76.1` F1, showing that interpretable-by-design models can still exploit contextual information rather than collapse to unigram heuristics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-transformer-2306-01128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-transformer-2306-01128]].
