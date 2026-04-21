---
type: concept
title: Code Comment
slug: code-comment
date: 2026-04-20
updated: 2026-04-20
aliases: [代码注释, source-code comment]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Code Comment** (代码注释) — natural-language text attached to source code that summarizes functionality, intent, or usage and can serve as semantic supervision for code understanding.

## Key Points

- SCodeR treats a comment `t` and its paired function `c` as a positive pair `(t, c)` during contrastive pre-training.
- The paper argues that comments describe function intent more directly than structure-preserving program transformations.
- Comment-code alignment helps tie together semantically similar implementations written in different programming languages.
- Gains on code search and notebook markdown ordering suggest that comment-aware positives improve joint code-text representations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-softlabeled-2210-09597]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-softlabeled-2210-09597]].
