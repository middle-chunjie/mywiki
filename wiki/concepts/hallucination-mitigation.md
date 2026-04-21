---
type: concept
title: Hallucination Mitigation
slug: hallucination-mitigation
date: 2026-04-20
updated: 2026-04-20
aliases: [anti-hallucination training, 幻觉缓解]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hallucination Mitigation** (幻觉缓解) — training or inference interventions designed to reduce unsupported model outputs, often by encouraging uncertainty expression or refusal when knowledge is absent.

## Key Points

- Phi-4 explicitly prefers refusal over fabricated answers on unknown questions and uses dedicated SFT and DPO data to teach that behavior.
- The report notes that this choice can lower benchmark F1 on SimpleQA even when it improves user-facing honesty.
- Despite these efforts, the authors still identify factual hallucination as a core unresolved weakness of the final model.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
