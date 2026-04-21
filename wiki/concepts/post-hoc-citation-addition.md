---
type: concept
title: Post-Hoc Citation Addition
slug: post-hoc-citation-addition
date: 2026-04-20
updated: 2026-04-20
aliases: [post-hoc citation generation, 后置引用添加]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Post-Hoc Citation Addition** (后置引用添加) — a pipeline that attaches citations to already generated text after generation rather than forcing the language model to emit citations during decoding.

## Key Points

- AGRaME instantiates this idea with PropCite, which decomposes generated sentences into propositions and scores source passages for each proposition.
- The method is model-agnostic at generation time because citation assignment happens after the answer has already been produced.
- Compared with prompt-based citation generation, PropCite consistently improves citation recall and often precision on ASQA and ELI5.
- The paper notes that lightweight proposition extraction alternatives, such as dependency-based claim decomposition, could reduce deployment cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[reddy-2024-agrame-2405-15028]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[reddy-2024-agrame-2405-15028]].
