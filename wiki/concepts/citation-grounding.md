---
type: concept
title: Citation Grounding
slug: citation-grounding
date: 2026-04-20
updated: 2026-04-20
aliases: [evidence-grounded citation, 引文证据对齐]
tags: [citations, evaluation, retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Citation Grounding** (引文证据对齐) — the requirement that claims in a generated answer be explicitly supported by localized cited evidence rather than by uncited or weakly linked sources.

## Key Points

- DR Tulu wraps claims with explicit citation tags that point to snippet-level evidence returned by search tools.
- The citation reward evaluates both format validity and per-claim evidence support, combining them into a final score.
- Citation grounding is central to the paper's notion of deep research because long-form answers must be auditable by humans.
- The paper reports large gains in citation precision and recall from RL with evolving rubrics, especially on SQAv2.
- The authors contrast DR Tulu with open baselines that usually provide no citations at all, which limits trustworthy verification.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shao-2025-dr-2511-19399]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shao-2025-dr-2511-19399]].
