---
type: concept
title: Closed-Domain Hallucination
slug: closed-domain-hallucination
date: 2026-04-20
updated: 2026-04-20
aliases: [context-based hallucination, 闭域幻觉]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Closed-Domain Hallucination** (闭域幻觉) — generation that contradicts or cannot be verified from the supplied context even though the answer should be grounded within that context.

## Key Points

- The paper distinguishes this from broader knowledge-based hallucination and treats it as a consequence of training on fragmented documents.
- Example failure modes include unsupported summary details, failure to follow swapped-answer context, and undefined names in generated code.
- Best-fit Packing reduces these failures by preserving grounding spans inside individual training samples whenever possible.
- The reported reduction reaches `58.3%` on undefined-name hallucinations for MBPP code generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-fewer-2404-10830]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-fewer-2404-10830]].
