---
type: concept
title: Semi-Parametric Language Model
slug: semi-parametric-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [semi-parametric LM, 半参数语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Semi-Parametric Language Model** (半参数语言模型) — a language model that combines parametric sequence modeling in weights with non-parametric memory accessed through retrieval at prediction time.

## Key Points

- RETRO is framed as a semi-parametric alternative to pure parameter scaling: knowledge can be stored in an external datastore instead of only in weights.
- The paper argues this decouples memorization from model size by allowing the model to access trillions of additional tokens through retrieval.
- Updating the retrieval database is much cheaper than retraining a frontier model, making the approach attractive for freshness and controllability.
- The work also shows the downside of semi-parametric memory: retrieval can expose leakage and enable direct copying from the training corpus.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[borgeaud-2022-improving-2112-04426]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[borgeaud-2022-improving-2112-04426]].
