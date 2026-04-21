---
type: concept
title: Natural Channel of Code
slug: natural-channel-of-code
date: 2026-04-20
updated: 2026-04-20
aliases: [natural channel, code natural channel]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Natural Channel of Code** — the human-facing channel of source code that conveys meaning through identifiers, comments, naming choices, and stylistic cues rather than formal execution semantics.

## Key Points

- [[jha-2023-codeattack-2206-00052]] builds directly on the dual-channel view of code and targets perturbations in the natural channel instead of the formal executable channel.
- The paper treats comments, variable names, function names, and related lexical cues as attack surfaces because PL models rely heavily on them.
- CodeAttack aims to keep adversarial edits fluent to a human reader while still degrading the victim model's generated output.
- The paper argues that robustness weaknesses in the natural channel reveal over-reliance on superficial lexical cues in current code models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jha-2023-codeattack-2206-00052]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jha-2023-codeattack-2206-00052]].
