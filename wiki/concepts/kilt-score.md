---
type: concept
title: KILT-Score
slug: kilt-score
date: 2026-04-20
updated: 2026-04-20
aliases: [KILT score, KILT-EM, KILT-AC, KILT-F1, KILT evaluation metric]
tags: [evaluation, retrieval, information-retrieval, benchmark]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**KILT-Score** — a composite evaluation metric for knowledge-intensive language tasks that multiplies R-Precision of retrieved provenance by the generation quality metric (EM, Accuracy, or F1), so that the score is zero unless the correct supporting document is retrieved.

## Key Points

- Formally: `KILT-M = (1/|Q|) Σ_{q∈Q} [RP(p, d)==1] · M(y, ŷ)`, where `RP` is R-Precision comparing retrieved passages to KILT provenance labels, and `M` is the downstream generation metric.
- The metric enforces that good generation scores cannot be achieved without grounding in the correct document, jointly measuring both retrieval quality and generation quality.
- Task-specific variants: KILT-EM for open-domain QA (NQ, TriviaQA, HotpotQA), KILT-AC for fact verification and slot-filling (FEVER, T-REx, zsRE), KILT-F1 for dialogue (WoW).
- Used as the utility function `U` in [[zamani-2024-stochastic]] during training, directly aligning the optimization objective with leaderboard evaluation; datasets with multiple provenance documents (HotpotQA, FEVER) use the union of provenance labels for RP computation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zamani-2024-stochastic]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zamani-2024-stochastic]].
