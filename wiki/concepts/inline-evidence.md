---
type: concept
title: Inline Evidence
slug: inline-evidence
date: 2026-04-20
updated: 2026-04-20
aliases: [inline evidence, 内联证据]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Inline Evidence** (内联证据) — a generation format in which a model emits an answer and a quoted evidence span inside the same output string using a structured syntax.

## Key Points

- [[menick-2022-teaching-2203-11147]] uses the template `"% <Claim> % (Document title) % [Quote] %"` so answer text and supporting quote are jointly generated.
- Because the answer appears before the evidence in the autoregressive ordering, the answer can be scored independently of the quote continuation.
- The format supports post-hoc parsing for UI rendering and online constrained decoding to guarantee verbatim quotation.
- The paper positions inline evidence as a practical way to reduce human verification effort compared with returning only a URL or an unsupported answer.
- The method still permits cherry-picked or misleading quotations, so inline evidence improves inspectability more than truth guarantees.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[menick-2022-teaching-2203-11147]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[menick-2022-teaching-2203-11147]].
