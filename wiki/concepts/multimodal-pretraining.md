---
type: concept
title: Multimodal Pre-training
slug: multimodal-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-modal pretraining, 多模态预训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multimodal Pre-training** (多模态预训练) — pre-training over multiple aligned modalities such as code, natural language, and program structure to learn shared representations.

## Key Points

- The survey divides CodePTMs into unimodal, bimodal, and multimodal systems depending on whether they use code alone or additional modalities.
- It argues that natural language in code and structured inputs such as ASTs and DFGs often improve code understanding relative to code-only inputs.
- The paper distinguishes `Together` fusion from `Standalone` fusion, noting that `Together` is better positioned to learn cross-modal representations.
- Cross-modal-aware pre-training tasks are further organized into code-NL, code-structure, and code-NL-structure subfamilies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[niu-2022-deep-2205-11739]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[niu-2022-deep-2205-11739]].
