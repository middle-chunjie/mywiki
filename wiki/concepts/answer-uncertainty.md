---
type: concept
title: Answer Uncertainty
slug: answer-uncertainty
date: 2026-04-20
updated: 2026-04-20
aliases: [uncertainty-aware answering, abstention in QA]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Answer Uncertainty** (答案不确定性) — a QA system's need to represent, detect, or act on insufficient evidence instead of confidently producing an unsupported answer.

## Key Points

- REALTIME QA adds a NOTA setting to test whether systems can cope with uncertainty rather than exploit multiple-choice heuristics.
- All baselines degrade noticeably under NOTA, showing that uncertainty handling remains weak even for the best open-book system.
- The paper highlights cases where GPT-3 keeps producing outdated answers when retrieval does not provide enough evidence.
- A central future-work direction is enabling the QA system to notify the user or trigger improved retrieval when evidence is insufficient.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kasai-2024-realtime-2207-13332]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kasai-2024-realtime-2207-13332]].
