---
type: concept
title: Truthfulness
slug: truthfulness
date: 2026-04-20
updated: 2026-04-20
aliases: [factual accuracy, 真实性]
tags: [llm, alignment]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Truthfulness** (真实性) — the property that a model's statements and judgments accurately reflect facts about the world rather than merely sounding plausible.

## Key Points

- [[kadavath-2022-language-2207-05221]] treats truthfulness as one part of a broader honesty agenda that also includes calibration and self-knowledge.
- The `P(True)` task operationalizes truthfulness at the answer level by asking whether a proposed answer is actually correct.
- The paper shows that truth-related judgments can use in-context evidence, because `P(IK)` increases when relevant source material is added to the prompt.
- The authors explicitly note that their setup still conflates "the truth" with "what humans say," leaving harder truthfulness settings unresolved.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kadavath-2022-language-2207-05221]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kadavath-2022-language-2207-05221]].
