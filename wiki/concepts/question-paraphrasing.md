---
type: concept
title: Question Paraphrasing
slug: question-paraphrasing
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt paraphrasing, query paraphrasing, 问题改写]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Question Paraphrasing** (问题改写) — rewriting an original question into a shorter semantically equivalent prompt that better matches the target interaction style or evaluation setting.

## Key Points

- InfiBench requires domain experts to paraphrase each selected Stack Overflow question into a succinct instruction closer to real code-LLM usage.
- The paper uses paraphrasing to reduce the style gap between verbose forum posts and direct developer prompts.
- Paraphrasing is also an anti-contamination measure because it weakens any advantage from memorizing the original Stack Overflow wording.
- The authors show that copying original Stack Overflow answers reaches only `65.18%`, supporting the claim that paraphrasing reduces memorization benefits.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-infibench-2404-07940]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-infibench-2404-07940]].
