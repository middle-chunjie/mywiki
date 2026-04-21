---
type: concept
title: Exact String Match
slug: exact-string-match
date: 2026-04-20
updated: 2026-04-20
aliases: [exact match, string exact match, 精确字符串匹配]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Exact String Match** (精确字符串匹配) — a strict sequence-level metric that gives full credit only when a predicted string exactly matches the target string with no token or character differences.

## Key Points

- [[schaeffer-2023-emergent-2304-15004]] names Exact String Match as one of the two metrics responsible for most reported BIG-Bench emergence cases.
- The paper argues that exact-match evaluation is nonlinear because longer outputs compound small token-level error rates into sharp drops in task accuracy.
- This metric can therefore make smooth scaling curves look abrupt and unpredictable.
- The paper contrasts it with smoother metrics such as [[token-edit-distance]], which preserve graded performance changes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[schaeffer-2023-emergent-2304-15004]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[schaeffer-2023-emergent-2304-15004]].
