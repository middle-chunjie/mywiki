---
type: concept
title: Explanation Tuning
slug: explanation-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [Explanation tuning, 解释调优]
tags: [llm, post-training, distillation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Explanation Tuning** (解释调优) — a post-training method that supervises a student model with explanation-rich teacher outputs, including reasoning steps and response structure, instead of only short final answers.

## Key Points

- [[mukherjee-2023-orca-2306-02707]] defines training examples as `⟨system message, user query, LFM response⟩`, explicitly expanding the supervision signal beyond standard instruction-response pairs.
- The method relies on longer teacher traces from [[chatgpt]] and [[gpt-4]], including step-by-step justifications, detailed answers, and response-format guidance.
- Orca combines explanation tuning with large-scale task diversity from FLAN-derived data rather than a narrow set of synthetic prompts or community chat logs.
- The paper frames staged training on ChatGPT then GPT-4 traces as a progressive-learning variant that makes explanation tuning more effective for a `13B` student.
- Empirically, the paper reports stronger reasoning gains than prior open 13B chat models, especially on AGIEval and Big-Bench Hard.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mukherjee-2023-orca-2306-02707]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mukherjee-2023-orca-2306-02707]].
