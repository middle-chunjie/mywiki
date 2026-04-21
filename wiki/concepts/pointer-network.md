---
type: concept
title: Pointer Network
slug: pointer-network
date: 2026-04-20
updated: 2026-04-20
aliases: [pointer networks, 指针网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pointer Network** (指针网络) — a sequence model that predicts positions in the input sequence instead of generating outputs only from a fixed vocabulary.

## Key Points

- This paper uses a multi-headed pointer network with one head for bug localization and one head for repair selection.
- Both heads define probability distributions over program-token positions, letting the model copy repair evidence from existing variable occurrences.
- The architecture uses LSTM hidden states plus attention `alpha = softmax(W^T M)` rather than a decoder that generates replacement tokens autoregressively.
- Bug-free classification is folded into the same mechanism by training the localization head to point to a special no-fault position.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[vasic-2019-neural-1904-01720]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[vasic-2019-neural-1904-01720]].
