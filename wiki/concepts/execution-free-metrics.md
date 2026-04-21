---
type: concept
title: Execution-Free Metrics
slug: execution-free-metrics
date: 2026-04-20
updated: 2026-04-20
aliases: [non-execution metrics, 不依赖执行的评测指标]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Execution-Free Metrics** (不依赖执行的评测指标) — code-generation metrics that score outputs by lexical, syntactic, or semantic similarity to references without running the generated program.

## Key Points

- [[wang-2023-executionbased-2212-10481]] compares BLEU, ROUGE, METEOR, ChrF, and CodeBLEU against execution-based evaluation.
- The paper finds that these metrics do not faithfully reproduce model rankings induced by pass@k and execution outcomes.
- BLEU and ROUGE correlate somewhat better with execution than the other metrics, but still do not distinguish passed and failed samples reliably.
- CodeBLEU is notably low in most settings studied, suggesting limited usefulness for snippet-style open-domain code evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-executionbased-2212-10481]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-executionbased-2212-10481]].
