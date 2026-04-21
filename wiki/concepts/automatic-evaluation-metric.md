---
type: concept
title: Automatic Evaluation Metric
slug: automatic-evaluation-metric
date: 2026-04-20
updated: 2026-04-20
aliases: [automatic metric, 自动评估指标]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Automatic Evaluation Metric** (自动评估指标) — a computable scoring function used to estimate generation quality without requiring fresh manual judgments for every output.

## Key Points

- [[cui-2022-codeexp-2211-15395]] evaluates code explanations with BLEU, ROUGE, METEOR, CER, BERTScore, and CodeBERTScore.
- The paper explicitly tests metric validity by comparing them against human preferences using [[kendalls-tau]].
- BLEU and METEOR are the most aligned with overall human judgments in the reported experiments.
- CER is introduced to better track coverage of code-relevant terms shared by source code, reference docstrings, and generated text.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cui-2022-codeexp-2211-15395]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cui-2022-codeexp-2211-15395]].
