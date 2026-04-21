---
type: concept
title: Self-Verification
slug: self-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [self verification, 自验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Verification** (自验证) — a procedure where a model checks the correctness of its own generated claims by producing and resolving targeted verification evidence before finalizing an answer.

## Key Points

- CoVe operationalizes self-verification by turning a draft answer into a set of verification questions and then answering them with the same underlying LLM.
- The paper shows that verification answers are often more accurate when asked independently than when embedded inside the original long-form response.
- Factored self-verification improves over joint verification because it avoids conditioning on previously hallucinated content.
- The approach works without retraining and without retrieval, making it a pure inference-time reliability intervention.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dhuliawala-2023-chainofverification-2309-11495]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dhuliawala-2023-chainofverification-2309-11495]].
