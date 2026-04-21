---
type: concept
title: Coarse-to-Fine Prompt Compression
slug: coarse-to-fine-prompt-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [coarse-to-fine compression, hierarchical prompt compression, 粗到细提示压缩]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Coarse-to-Fine Prompt Compression** (粗到细提示压缩) — a two-stage compression strategy that first removes larger prompt units under a global budget and then refines the retained content at token level.

## Key Points

- [[jiang-2023-llmlingua-2310-05736]] first performs coarse-grained demonstration-level filtering before running fine-grained token selection.
- The coarse stage preserves semantic integrity better than immediately dropping isolated tokens at extreme compression ratios.
- Demonstration filtering is driven by small-LM perplexity ranking and constrained by `k · τ_dems · L_dems`.
- The fine stage uses iterative conditional scoring so that later segment compression sees earlier retained compressed content.
- The paper attributes its strong performance at `14x-20x` compression partly to this staged design.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2023-llmlingua-2310-05736]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2023-llmlingua-2310-05736]].
