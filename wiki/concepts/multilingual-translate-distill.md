---
type: concept
title: Multilingual Translate-Distill
slug: multilingual-translate-distill
date: 2026-04-20
updated: 2026-04-20
aliases: [MTD, Multilingual TD, 多语言翻译蒸馏]
tags: [retrieval, distillation, multilingual]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multilingual Translate-Distill** (多语言翻译蒸馏) — a multilingual retrieval training framework that distills monolingual teacher scores into a multilingual student by translating training passages into all target document languages.

## Key Points

- [[yang-2024-distillation-2405-00977]] generalizes Translate-Distill from one target document language to MLIR, where documents from several languages must be scored on a common scale.
- The method uses an English selector and scorer to produce teacher supervision, then translates each sampled passage into all target languages before student optimization.
- The student is trained with a `KL` objective over teacher and student scores rather than direct relevance labels, so the pipeline can start from unlabeled English query-passage data.
- The paper studies three mini-batch language assignment schemes and shows that MTD remains effective across them as long as multiple languages appear in each batch.
- MTD beats `ColBERT-X MTT` on all reported collections, with especially large improvements on NeuCLIR.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-distillation-2405-00977]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-distillation-2405-00977]].
