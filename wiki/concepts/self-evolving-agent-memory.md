---
type: concept
title: Self-Evolving Agent Memory
slug: self-evolving-agent-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [self-evolving memory, adaptive agent memory, 自演化代理记忆]
tags: [agents, memory, llm, self-evolving]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Evolving Agent Memory** (自演化代理记忆) — a memory management paradigm in which both the policy governing memory construction and the set of memory operations themselves are improved from interaction experience, with minimal manual supervision.

## Key Points

- MemSkill implements a closed-loop optimization that alternates between (i) training the controller to select skills from the current bank, and (ii) using the designer to evolve the skill bank itself from hard cases mined during training.
- The key distinction from prior work (e.g., Memory-R1, Mem-α) is that MemSkill evolves the *skill bank operations* rather than just optimizing a fixed memory policy.
- Concurrent work Evo-Memory targets test-time streaming benchmarks, while MemEvolve meta-optimizes memory architectures within a predefined modular space; MemSkill instead focuses on evolving the reusable memory skills directly.
- Empirical evidence shows strong cross-model and cross-dataset generalization: skills trained on LLaMA transfer to Qwen without retraining, and skills learned on LoCoMo transfer to LongMemEval and HotpotQA.
- The designer prevents regressions via snapshot rollback and early stopping, making the evolution process robust to noisy updates.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2026-memskill-2602-02474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2026-memskill-2602-02474]].
