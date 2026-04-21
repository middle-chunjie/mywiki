---
type: entity
title: TIFA
slug: tifa
date: 2026-04-20
entity_type: dataset
aliases: [TIFA, TIFA v1.0]
tags: []
---

## Description

TIFA is the open-ended text-to-image evaluation dataset used in [[cho-2023-visual-2305-15328]] for `TIFA160` prompt evaluation and human-correlation analysis. The paper also adapts TIFA in-context prompts when asking `GPT-3.5-Turbo` to synthesize evaluation programs.

## Key Contributions

- Supplies `160` open-ended prompts and associated human judgments for evaluation.
- Provides the open-ended setting in which VPEval compares against TIFA with BLIP-2.
- Serves as the source of in-context prompt-program exemplars for program generation.

## Related Concepts

- [[text-to-image-generation]]
- [[in-context-learning]]
- [[explainable-evaluation]]

## Sources

- [[cho-2023-visual-2305-15328]]
