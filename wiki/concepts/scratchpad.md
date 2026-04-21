---
type: concept
title: ScratchPad
slug: scratchpad
date: 2026-04-20
updated: 2026-04-20
aliases: [scratch pad, scratchpad prompting]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**ScratchPad** (草稿板推理) — a prompting style where intermediate computational states are written out explicitly so a model can reason through multi-step procedures step by step.

## Key Points

- The paper cites ScratchPad as an important precursor because it already shows that language models can simulate code execution through state traces.
- In the authors' framing, ScratchPad corresponds to LM-side execution with explicit intermediate state updates.
- Chain of Code inherits the state-trace idea but combines it with real interpreter execution rather than relying on language-model simulation alone.
- The ablations show that explicit LM state tracking is better than LM-only final-answer generation, supporting the underlying ScratchPad intuition.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-chain-2312-04474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-chain-2312-04474]].
