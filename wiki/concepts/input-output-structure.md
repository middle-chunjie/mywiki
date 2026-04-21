---
type: concept
title: Input-Output Structure
slug: input-output-structure
date: 2026-04-20
updated: 2026-04-20
aliases: [IO structure, input-output specification block, 输入输出结构]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Input-Output Structure** (输入输出结构) — an explicit specification of a program's input parameters, parameter types, and expected outputs used to anchor reasoning and implementation.

## Key Points

- SCoT prepends an input-output block before the solving process so the model identifies the code's entry and exit conditions early.
- The paper argues this helps the model understand task requirements before planning loops or branches.
- Removing the input-output structure causes smaller but consistent accuracy drops across HumanEval, MBPP, and MBCPP.
- The authors show examples where omitting this block leads the model to return the wrong output format even when most of the logic is correct.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structured-2305-06599]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structured-2305-06599]].
