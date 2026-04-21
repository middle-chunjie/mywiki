---
type: concept
title: Generative Error Correction
slug: generative-error-correction
date: 2026-04-20
updated: 2026-04-20
aliases: [GER, 生成式错误纠正]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Generative Error Correction** (生成式错误纠正) — an ASR post-processing formulation in which a model generates the target transcript directly from an `N`-best hypothesis list instead of merely reranking candidates.

## Key Points

- The paper frames GER as learning a hypotheses-to-transcription mapping `Y = M_H2T(Y_N)` with autoregressive cross-entropy supervision.
- Compared with LM rescoring, GER can exploit all candidates in the `N`-best list and is not limited to selecting one existing hypothesis.
- In noisy settings, vanilla GER still underperforms because the linguistic input remains corrupted by source audio noise.
- RobustGER extends GER by adding an explicit denoising signal derived from hypothesis diversity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2401-10446]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2401-10446]].
