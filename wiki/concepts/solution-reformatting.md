---
type: concept
title: Solution Reformatting
slug: solution-reformatting
date: 2026-04-20
updated: 2026-04-20
aliases: [solution reformatting, solution segmentation, step segmentation normalization, 解题步骤重格式化]
tags: [data-preprocessing, benchmark-construction, llm]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Solution Reformatting** (解题步骤重格式化) — a preprocessing step in benchmark construction that standardises the granularity of reasoning steps in LLM-generated solutions by using a model to re-segment paragraphs at logically complete boundaries, without altering solution content.

## Key Points

- Raw LLM-generated solutions have inconsistent step granularity: some models use many fine-grained double-linebreak segments yielding logically incomplete steps; others produce monolithic paragraphs combining multiple logical components.
- [[processbench]] implements this by: (1) collapsing all linebreaks to whitespace, (2) asking Qwen2.5-72B-Instruct to re-insert `\n\n` at logical step boundaries, (3) discarding solutions whose final answers change post-reformatting (< 0.5% rate).
- The goal is to ensure double-linebreak segments correspond to "logically complete and progressive reasoning steps," making human annotation of step correctness tractable and consistent.
- A critical quality control: even with explicit instructions to preserve content, model-based reformatting introduces a small content alteration rate that must be filtered.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zheng-2024-processbench-2412-06559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zheng-2024-processbench-2412-06559]].
