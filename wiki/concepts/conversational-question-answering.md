---
type: concept
title: Conversational Question Answering
slug: conversational-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [CQA, conversational QA, 对话式问答]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Conversational Question Answering** (对话式问答) — a question answering setting in which each turn must be answered from a context while conditioning on the prior question-answer history.

## Key Points

- The paper formalizes CQA with context `C` and dialogue history `H_{<t}`, making each answer depend on earlier turns rather than on a standalone question only.
- It argues that limited multi-turn supervision is a central bottleneck for stronger CQA systems.
- S2M converts single-turn QA data into synthetic conversations to create additional CQA training resources.
- On QuAC, S2M dramatically improves unsupervised performance, raising DeBERTa from `13.2` to `59.2` F1.
- After supervised finetuning on QuAC, the best S2M-based system reaches `76.4` dev F1 and `76.3` test F1.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-sm-2312-16511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-sm-2312-16511]].
