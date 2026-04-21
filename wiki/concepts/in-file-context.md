---
type: concept
title: In-file Context
slug: in-file-context
date: 2026-04-20
updated: 2026-04-20
aliases: [Local Context, 文件内上下文]
tags: [code-completion]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**In-file Context** (文件内上下文) — the prefix tokens within the current source file that a causal code model can condition on when predicting the next code token.

## Key Points

- The paper defines the in-file context at time step `t` as `s_t = (x_1, ..., x_{t-1})`.
- Standard CodeGen baselines rely only on in-file context and therefore miss project-specific APIs defined elsewhere.
- CoCoMIC preserves causal in-file modeling and augments it with cross-file entity representations rather than replacing the local prefix.
- In-file context alone already covers `75.19%` of ground-truth identifiers on the benchmark, but still leaves substantial project-level gaps.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2023-cocomic-2212-10007]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2023-cocomic-2212-10007]].
