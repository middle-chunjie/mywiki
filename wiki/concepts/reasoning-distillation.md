---
type: concept
title: Reasoning Distillation
slug: reasoning-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [推理蒸馏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reasoning Distillation** (推理蒸馏) — the transfer of reasoning behavior from a stronger teacher model to a student model by training on teacher-generated reasoning traces or demonstrations.

## Key Points

- This paper distills long chain-of-thought reasoning from `DeepSeek-R1` and `QwQ-32B-Preview` into `Qwen2.5-32B-Instruct`.
- The transfer is data-efficient: `17k` demonstrations are enough for large gains on math and coding benchmarks.
- The transfer is also parameter-efficient: LoRA with small trainable adapters remains competitive with full supervised fine-tuning.
- The paper argues that the most important signal is not step-level factual correctness alone, but the structural organization of the reasoning trace.
- Distilled models retain much more non-reasoning capability than directly using a reasoning-specialized teacher such as QwQ.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-llms-2502-07374]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-llms-2502-07374]].
