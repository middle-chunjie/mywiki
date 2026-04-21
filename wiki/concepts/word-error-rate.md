---
type: concept
title: Word Error Rate
slug: word-error-rate
date: 2026-04-20
updated: 2026-04-20
aliases: [WER, 词错误率]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Word Error Rate** (词错误率) — the standard ASR evaluation metric that measures transcript errors through insertions, deletions, and substitutions relative to a reference transcription.

## Key Points

- The paper reports all major results in WER and uses relative WER reduction as its main improvement metric.
- The token-level language-space noise embedding is explicitly designed to correspond to edit-distance-like token differences and therefore to WER behavior.
- RobustGER achieves its headline result by reducing CHiME-4 average WER from `12.8` to `5.9`, a `53.9%` relative reduction.
- The paper also reports `N`-best oracle and compositional oracle WERs to characterize the upper bounds of reranking and token-constrained GER.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2401-10446]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2401-10446]].
