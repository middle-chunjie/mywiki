---
type: concept
title: Privacy Leakage
slug: privacy-leakage
date: 2026-04-20
updated: 2026-04-20
aliases: [information leakage, 隐私泄漏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Privacy Leakage** (隐私泄漏) — unintended disclosure of sensitive information from a model, representation, or system output that was assumed to be safer than the underlying raw data.

## Key Points

- The paper argues that dense text embeddings leak enough signal to permit near-exact text reconstruction.
- On pseudo re-identified MIMIC notes, Vec2Text recovers `89.2%` of full names, making the privacy risk directly measurable rather than hypothetical.
- The results imply that embedding-only storage does not reliably anonymize text for third-party vector database services.
- Small Gaussian perturbations reduce reconstruction quality sharply, but stronger perturbations also damage retrieval utility, so privacy protection is not free.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[morris-2023-text-2310-06816]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[morris-2023-text-2310-06816]].
