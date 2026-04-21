---
type: concept
title: Logit Masking
slug: logit-masking
date: 2026-04-20
updated: 2026-04-20
aliases: [token masking]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Logit masking** — the operation of modifying next-token logits so invalid tokens are suppressed before sampling.

## Key Points

- In the paper, the mask is generated from monitor state and combined with logits through `ℓ ⊕ m`.
- Invalid tokens are reset to a large negative constant `-K`, while admissible tokens keep their original logits.
- Because identifiers may span multiple subtokens, masking is prefix-aware rather than limited to full identifier strings.
- The technique allows a black-box or frozen LM to respect external symbolic constraints at decoding time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[agrawal-nd-monitorguided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[agrawal-nd-monitorguided]].
