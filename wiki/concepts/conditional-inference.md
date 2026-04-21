---
type: concept
title: Conditional Inference
slug: conditional-inference
date: 2026-04-20
updated: 2026-04-20
aliases: [metadata-conditioned inference]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Conditional Inference** — an inference-time procedure that prepends a control string, such as metadata, to the prompt in order to steer a language model's behavior without changing its weights.

## Key Points

- MeCo enables URL-conditioned prompting after pre-training, even when the model is also usable without metadata thanks to cooldown.
- The paper designs task-specific URLs such as `www.factquizmaster.com` and `www.socialskillsassessment.com` to improve zero-shot or few-shot performance.
- On the main `1.6B` model, conditional inference improves the MeCo average from `56.7` to `57.2`, while giving almost no gain to a standard model.
- Real URLs with different connotations matter: `www.factmonster.com` outperforms `boards.4chan.org` by large margins on multiple QA tasks.
- Conditioning on `en.wikipedia.org` also lowers harmful-generation toxicity scores more strongly for MeCo than for the baseline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2025-metadata-2501-01956]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2025-metadata-2501-01956]].
