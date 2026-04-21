---
type: concept
title: Metadata Conditioning
slug: metadata-conditioning
date: 2026-04-20
updated: 2026-04-20
aliases: [metadata-conditioned training]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Metadata Conditioning** — a pre-training strategy that prepends document-level metadata to model inputs so the model can exploit source or topic signals while predicting the document text.

## Key Points

- MeCo prepends absolute domain names such as `en.wikipedia.org` before each document during the first `90%` of pre-training.
- The loss is computed only on document tokens, so metadata acts as conditioning context rather than a direct language-modeling target.
- The paper shows that semantically meaningful metadata is not strictly required: hashed URLs perform as well as readable URLs.
- Gains persist across `600M` to `8B` models and across `C4`, `RefinedWeb`, and `DCLM`, suggesting the method is robust to scale and corpus choice.
- The authors argue that metadata conditioning helps group related documents together, improving data efficiency and later prompt-time steerability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2025-metadata-2501-01956]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2025-metadata-2501-01956]].
