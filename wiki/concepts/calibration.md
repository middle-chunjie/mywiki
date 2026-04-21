---
type: concept
title: Calibration
slug: calibration
date: 2026-04-20
updated: 2026-04-20
aliases: [probability calibration, 校准]
tags: [llm, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Calibration** (校准) — the property that a model's predicted probabilities match the empirical frequencies with which events are actually correct.

## Key Points

- [[kadavath-2022-language-2207-05221]] shows that explicit letter-labeled answer options yield much better calibration than default free-form BIG-Bench formatting.
- Calibration improves with model size from `800M` to `52B` and also improves when evaluation moves from zero-shot to few-shot prompting.
- Replacing a multiple-choice option with "none of the above" harms both accuracy and calibration, even for strong models.
- RLHF policies appear miscalibrated at first, but the paper shows that temperature scaling with `T = 2.5` can partly restore calibration.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kadavath-2022-language-2207-05221]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kadavath-2022-language-2207-05221]].
