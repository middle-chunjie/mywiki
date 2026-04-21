---
type: concept
title: Semantic Mismatch
slug: semantic-mismatch
date: 2026-04-20
updated: 2026-04-20
aliases: [语义失配]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Semantic Mismatch** (语义失配) — the failure mode where the same surface token refers to different meanings across contexts, causing context-insensitive retrieval signals to overestimate or underestimate relevance.

## Key Points

- COIL treats semantic mismatch as a core weakness of classical lexical retrieval, separate from vocabulary mismatch.
- The motivating examples include polysemous tokens such as "bank" and context-dependent stop words such as "is".
- Contextualized token encoders are used specifically to differentiate same-form tokens under different local meanings.
- The paper's case studies show large score gaps between relevant and irrelevant uses of the same token, such as "cabinet" in government versus furniture contexts.
- The authors claim that addressing semantic mismatch is what lets exact-match retrieval recover much of the fine-grained interaction previously associated with neural models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-coil-2104-07186]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-coil-2104-07186]].
