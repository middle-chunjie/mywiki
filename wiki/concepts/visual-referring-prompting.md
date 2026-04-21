---
type: concept
title: Visual Referring Prompting
slug: visual-referring-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [visual referring prompting, 视觉指代提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Visual Referring Prompting** (视觉指代提示) — a prompting method that edits image pixels with visual markers or scene text so the model can follow grounded instructions directly in visual space.

## Key Points

- The paper introduces this as a new interaction mechanism built on GPT-4V's ability to understand arrows, circles, boxes, and handwritten annotations on images.
- It is presented as more reliable than pure coordinate prompts in the reported experiments, especially for grounded description and document or chart interaction.
- The method supports tasks such as pointing to queried regions, attaching indices to objects, asking questions near visual structures, and expressing abstract patterns on an image.
- The report further explores a closed loop where GPT-4V can generate approximate pointing outputs itself for later interpretation or multi-step reasoning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-dawn-2309-17421]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-dawn-2309-17421]].
