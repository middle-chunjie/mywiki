---
type: concept
title: PassRatio
slug: pass-ratio
date: 2026-04-20
updated: 2026-04-20
aliases: [test pass ratio, 测试通过比例]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**PassRatio** (测试通过比例) — the fraction of test cases that a generated program passes, yielding a continuous score in `[0,1]` as an estimate of functional correctness.

## Key Points

- [[dong-2023-codescore-2301-09043]] defines `PassRatio = (1 / |C_p|) Σ 𝕀{Eval(g_p, I_{p,c}) = O_{p,c}}`.
- The paper chooses PassRatio instead of binary pass/fail because continuous supervision captures finer execution similarity between candidate programs.
- CodeScore is directly trained to regress toward PassRatio through a squared-error objective.
- Extended test suites are used to produce more reliable PassRatio labels for APPS-Eval, MBPP-Eval, and HE-Eval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2023-codescore-2301-09043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2023-codescore-2301-09043]].
