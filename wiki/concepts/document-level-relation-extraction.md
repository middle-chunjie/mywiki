---
type: concept
title: Document-level Relation Extraction
slug: document-level-relation-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [DocRE, document level relation extraction, 文档级关系抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Document-level Relation Extraction** (文档级关系抽取) — the task of identifying semantic relations between entities using evidence distributed across an entire document rather than a single sentence.

## Key Points

- [[sun-2024-consistency-2401-13598]] frames document-level relation extraction as materially harder than sentence-level extraction because relevant evidence can span multiple sentences and discourse structures.
- The paper evaluates zero-shot document-level relation extraction on [[docred]] and [[re-docred]] by splitting relation types into seen and unseen sets.
- GenRDK generates full synthetic documents, entities, and relation labels so that the same synthetic corpus can supervise both ZeroDocRE and ZeroDocRTE.
- For ZeroDocRE, the paper uses a graph-based DocRE backbone with RoBERTa-large and reports test `F1` up to `41.3 ± 8.9` on Re-DocRED and `41.5 ± 8.7` on DocRED at `m = 5`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-consistency-2401-13598]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-consistency-2401-13598]].
