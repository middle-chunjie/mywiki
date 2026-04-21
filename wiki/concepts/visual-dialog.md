---
type: concept
title: Visual Dialog
slug: visual-dialog
date: 2026-04-20
updated: 2026-04-20
aliases: [VisDial, 视觉对话]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Visual Dialog** (视觉对话) — a multimodal interaction setting where a system and a user exchange multiple rounds of questions and answers about an image.

## Key Points

- The paper repurposes the VisDial dataset for retrieval by treating each dialog's underlying image as the retrieval target.
- This reuse avoids collecting a new large-scale retrieval-specific conversation dataset, which the authors describe as costly and cumbersome.
- BLIP2 is used as an automatic answerer in many experiments to simulate the visual-dialog response process at scale.
- The paper's real-user study then checks how far this automatic proxy diverges from human answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[levy-nd-chatting]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[levy-nd-chatting]].
