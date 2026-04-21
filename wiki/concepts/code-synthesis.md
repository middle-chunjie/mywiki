---
type: concept
title: Code Synthesis
slug: code-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [Code Synthesis, 代码合成, Program Synthesis]
tags: [code, generation, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Synthesis** (代码合成) — the task of generating executable code from a specification, description, or contextual prompt.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] instantiates RTC for synthesis by describing a code region and regenerating it from that description.
- In SYNTHESISRTC, the target region is replaced with a TODO plus the forward model's natural-language description.
- The paper uses unit-test success as the semantic proxy for correctness.
- HumanEval and ARCADE are used as narrow-domain calibration settings for this task.
- Cross-domain synthesis evaluation on open-source repositories reveals substantial performance variation across projects.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
