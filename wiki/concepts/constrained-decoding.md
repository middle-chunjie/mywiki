---
type: concept
title: Constrained Decoding
slug: constrained-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [controlled decoding]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Constrained decoding** (约束解码) — a generation procedure that restricts the set of admissible next tokens so outputs satisfy structural or semantic constraints.

## Key Points

- MGD is an instance of constrained decoding driven by semantic constraints from static analysis rather than only syntax.
- The paper applies constraints by combining LM logits with a token mask during decoding, leaving the base model frozen.
- The authors position MGD as complementary to prompt-side conditioning approaches because it acts on the output distribution instead of the input.
- Richer constraints explored in the paper include valid enum cases, argument arity, typestates, and session-type protocols.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[agrawal-nd-monitorguided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[agrawal-nd-monitorguided]].
