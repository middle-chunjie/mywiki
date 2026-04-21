---
type: concept
title: Token Classification
slug: token-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [Token Classification, 词元分类, Token-Level Classification]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Token Classification** (词元分类) — the task of predicting a label for each token or span in a sequence using both local cues and contextual information.

## Key Points

- [[li-2024-chulo-2410-11119]] uses token classification to test whether compressed chunk representations still preserve fine-grained evidence.
- For token-level tasks, ChuLo feeds chunk representations into a BERT-decoder module so predictions can use document-wide context rather than a truncated window.
- On CoNLL-2012 and GUM, the method reports `0.9334` and `0.9555` micro-F1, respectively, outperforming Longformer and BigBird.
- The paper emphasizes that token classification is especially sensitive to information loss because every token may depend on context outside a local chunk.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-chulo-2410-11119]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-chulo-2410-11119]].
