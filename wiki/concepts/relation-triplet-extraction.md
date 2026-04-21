---
type: concept
title: Relation Triplet Extraction
slug: relation-triplet-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [RTE, relation triplet extraction, 关系三元组抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Relation Triplet Extraction** (关系三元组抽取) — the task of jointly extracting a head entity, a tail entity, and their semantic relation from text.

## Key Points

- [[sun-2024-consistency-2401-13598]] extends relation triplet extraction to a zero-shot document-level setting where target relation types are unseen during supervised training.
- The paper formulates ZeroDocRTE outputs as triplets `(head entity, tail entity, unseen relation type)` extracted from whole documents rather than isolated sentences.
- GenRDK asks ChatGPT to generate not only documents but also entity sets, relation triplets, reasoning explanations, and support sentences before assembling final labels.
- The denoised synthetic corpus is used to fine-tune LLaMA2-13B-Chat for final extraction, reaching test `F1` of `13.1 ± 2.6` on Re-DocRED and `14.2 ± 1.3` on DocRED at `m = 5`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-consistency-2401-13598]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-consistency-2401-13598]].
