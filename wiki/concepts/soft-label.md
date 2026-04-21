---
type: concept
title: Soft Label
slug: soft-label
date: 2026-04-20
updated: 2026-04-20
aliases: [软标签, relevance score target]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Soft Label** (软标签) — a non-binary supervision signal, typically a probability or graded relevance score, that encodes degrees of similarity rather than a hard positive/negative decision.

## Key Points

- SCodeR uses discriminator-predicted relevance scores between samples as soft labels for contrastive pre-training.
- These soft labels are produced separately for text-code pairs and code-code pairs through two discriminators.
- The adversarial loss reweights negatives by their soft relevance, so semantically similar negatives are not pushed away as aggressively as unrelated ones.
- A distillation objective also matches the dual encoder's output distribution to the discriminator's soft-label distribution.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-softlabeled-2210-09597]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-softlabeled-2210-09597]].
