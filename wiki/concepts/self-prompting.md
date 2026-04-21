---
type: concept
title: Self-Prompting
slug: self-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [self prompting, 自提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Prompting** (自提示) — a prompting strategy in which a language model generates auxiliary examples, rationales, or demonstrations that are then reused to improve its own downstream inference.

## Key Points

- SP-CoT explicitly frames itself as an LLM-only self-prompting framework: the model generates the training-style QA quadruplets, constructs demonstrations, and consumes them again at inference time.
- The method adds stronger quality control than prior self-prompting work by double-checking generated answers and enforcing composability constraints during question-chain construction.
- The generated demonstrations are not static; they are adaptively selected per input question through clustering and similarity retrieval.
- The paper shows that this self-prompting pipeline is especially helpful for smaller instruction-tuned models, where naive zero-shot prompting performs poorly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfprompted-2310-13552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfprompted-2310-13552]].
