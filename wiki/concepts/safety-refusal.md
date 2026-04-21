---
type: concept
title: Safety Refusal
slug: safety-refusal
date: 2026-04-20
updated: 2026-04-20
aliases: [refusal behavior, 安全拒答]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Safety Refusal** (安全拒答) — the behavior of declining, redirecting, or otherwise withholding substantial assistance on unsafe or disallowed user requests.

## Key Points

- SORRY-Bench operationalizes refusal as the complement of fulfillment on unsafe instruction-response pairs.
- A response can still count as fulfillment even if it begins with a disclaimer, as long as it provides substantial help for the unsafe task.
- The paper studies refusal across `44` fine-grained safety categories rather than only broad harm domains.
- Refusal rates vary sharply by model family, model version, and prompt mutation, showing that safety refusal is highly context-sensitive.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2024-sorrybench-2406-14598]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2024-sorrybench-2406-14598]].
