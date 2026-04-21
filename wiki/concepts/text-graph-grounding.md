---
type: concept
title: Text-Graph Grounding
slug: text-graph-grounding
date: 2026-04-20
updated: 2026-04-20
aliases: [text graph grounding]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Text-Graph Grounding** — a cross-modal alignment procedure that maps graph structural representations and node text representations into a shared space so language models can consume graph information more faithfully.

## Key Points

- GraphGPT encodes graph structure with a pretrained graph encoder and node text with a transformer-style text encoder, then normalizes both representations before alignment.
- The paper uses a contrastive objective `L = sum_i 0.5 * lambda_i * [CE(Gamma_i, y) + CE(Gamma_i^T, y)]` to align graph-side and text-side embeddings.
- The grounding step is the bridge between raw graph structure and the later instruction-tuned LLM, rather than a downstream prediction module by itself.
- The authors argue this stage reduces the need to serialize large neighborhoods into long textual prompts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-graphgpt-2310-13023]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-graphgpt-2310-13023]].
