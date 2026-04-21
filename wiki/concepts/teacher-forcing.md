---
type: concept
title: Teacher Forcing
slug: teacher-forcing
date: 2026-04-20
updated: 2026-04-20
aliases: [教师强制]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Teacher Forcing** (教师强制) — a training regime for autoregressive sequence models in which ground-truth previous tokens, rather than model-generated ones, are fed back as context during learning.

## Key Points

- The paper frames standard next-token language-model pretraining as teacher-forced optimization over local one-step predictions.
- It argues that teacher forcing creates a mismatch with autoregressive inference, where early mistakes can derail later generations.
- Multi-token prediction is presented as a partial remedy because each hidden state must support several future targets instead of only the immediate next one.
- The authors further claim that under MTP, semantically important choice points receive larger effective weight in the loss than inconsequential transitions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gloeckle-2024-better-2404-19737]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gloeckle-2024-better-2404-19737]].
