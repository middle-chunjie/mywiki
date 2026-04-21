---
type: concept
title: Context-Level FIM
slug: context-level-fim
date: 2026-04-20
updated: 2026-04-20
aliases: [context level FIM, 上下文级FIM]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context-Level FIM** (上下文级FIM) — a fill-in-the-middle augmentation strategy that applies the transformation after document packing and chunking, inside each training context.

## Key Points

- The method first splits a packed context around `<EOT>`, turns selected documents into FIM examples, rejoins them, and then trims or pads back to context length.
- It avoids many broken examples that occur when document-level augmentation is later fragmented by packing and chunking.
- The paper reports consistent sampling gains over document-level FIM even when the reduction in FIM loss is very small.
- Context-level FIM is the paper's preferred implementation despite being more involved operationally.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bavarian-2022-efficient-2207-14255]].
