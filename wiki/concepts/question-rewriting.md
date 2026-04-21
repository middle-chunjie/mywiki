---
type: concept
title: Question Rewriting
slug: question-rewriting
date: 2026-04-20
updated: 2026-04-20
aliases: [QR, conversational question rewriting, 问题改写]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Question Rewriting** (问题改写) — the transformation of a question into a different surface form while preserving intent, often to add or remove conversational dependence.

## Key Points

- The paper reverses the usual rewriting direction by converting self-contained questions into context-dependent follow-up questions.
- It constructs R-CANARD from CANARD using tuples `(C, H_{<t}, q_t, q_t^c)` and trains a reverse conversational question rewriter.
- The rewriting model is optimized as `p_q^{CQR}(q_t^c | C, H_{<t}, q_t)`.
- Rewriting improves dialogue naturalness by replacing repeated mentions with pronouns and dropping already-shared context.
- This stage is necessary because graph-reassembled QA sequences are coherent but not yet conversational in style.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-sm-2312-16511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-sm-2312-16511]].
