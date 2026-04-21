---
type: concept
title: Claim Extraction
slug: claim-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [claim decomposition, 断言抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Claim Extraction** (断言抽取) — the automatic decomposition of long-form text into self-contained verifiable claims suitable for downstream grounding or evaluation.

## Key Points

- PIC evaluation first extracts claims from model outputs so faithfulness can be scored at a finer granularity than whole-response judgments.
- The appendix adopts the extraction procedure of Song et al., using few-shot prompting with a sliding-window context over neighboring sentences.
- Extracted claims are deduplicated before verification to avoid inflated scores from repeated content.
- Claim extraction is also used to transform existing long-form instruction-tuning data into PIC-formatted training examples.
- Human evaluation reports strong agreement with the extractor on faithfulness, decontextualization, and quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2025-precise-2506-06589]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2025-precise-2506-06589]].
