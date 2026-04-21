---
type: concept
title: Reproduction Test Generation
slug: reproduction-test-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [reproduction tests, 复现测试生成]
tags: [software-engineering, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reproduction Test Generation** (复现测试生成) — generating executable tests that reproduce a reported bug before a fix and can confirm that the bug is resolved after patching.

## Key Points

- Agentless Mini uses sampled reproduction tests to rerank candidate patches for SWE-bench evaluation.
- The scaffold can retrieve a relevant existing test file to guide the model when generating reproduction tests.
- Unlike earlier Agentless settings that keep a single majority test, Agentless Mini can retain multiple top test samples.
- The paper's scaling study shows more reproduction-test samples improve final resolve rate up to a saturation point.
- Although SWE-RL trains only on repair editing, the resulting model generalizes to reproduction-test generation at inference time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wei-2025-swerl-2502-18449]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wei-2025-swerl-2502-18449]].
