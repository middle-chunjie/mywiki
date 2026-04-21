---
type: concept
title: Structural image transcription
slug: structural-image-transcription
date: 2026-04-20
updated: 2026-04-20
aliases: [structured image transcription]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Structural image transcription** — the task of converting an image whose content follows explicit spatial or grammatical structure into a symbolic token sequence while preserving both symbols and their organization.

## Key Points

- The paper treats music scores, formulas, and multi-line structured text as representative structural images rather than plain OCR inputs.
- Correct transcription requires not only recognizing local objects but also following a reading path consistent with the image's internal grammar.
- STN models the task with a two-stage process: localize a spotlighted region first, then generate the next output token.
- The paper evaluates this setting on Melody, Formula, and Multi-Line datasets and shows that explicit structure-aware focusing improves token accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yin-2018-transcribing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yin-2018-transcribing]].
