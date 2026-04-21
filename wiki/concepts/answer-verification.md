---
type: concept
title: Answer Verification
slug: answer-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [verification, 答案验证]
tags: [reasoning, retrieval, factuality]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Answer Verification** (答案验证) — the process of checking a model-produced intermediate answer against retrieved evidence and correcting it only when evidence confidence is sufficiently high.

## Key Points

- For each query-answer node `(q_i, a_i)`, SearChain retrieves the Top-1 document `d_i` and uses a reader to extract answer `g` and confidence `f`.
- On short-form tasks, consistency is checked by whether `g` appears in `a_i`; on long-form tasks, the paper uses a ROUGE threshold `alpha = 0.35`.
- The system only overrides the model when the retrieved evidence disagrees with the answer and `f > theta`, with `theta = 1.5` chosen to reduce retrieval-induced errors.
- Verification is one of the main reasons SearChain lowers the rate at which IR misleads the LLM compared with Self-Ask, ReAct, DSP, and related baselines.
- Removing verification causes large accuracy drops, including HotpotQA `56.91 -> 46.11` and zsRE `57.29 -> 43.58`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-searchinthechain-2304-14732]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-searchinthechain-2304-14732]].
