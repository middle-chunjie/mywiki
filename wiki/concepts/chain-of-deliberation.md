---
type: concept
title: Chain-of-Deliberation
slug: chain-of-deliberation
date: 2026-04-20
updated: 2026-04-20
aliases:
  - CoD
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chain-of-Deliberation** — a latent reasoning procedure that inserts multiple intermediate thinking steps before the final representation is produced, allowing a model to iteratively refine an embedding rather than emitting it in one pass.

## Key Points

- Debater applies Chain-of-Deliberation only to document encoding, appending prompt tokens `t_1, ..., t_m` so the model generates a sequence of step-specific document embeddings.
- The paper scores every thinking step against the query and uses `f_max(q, d) = max_i sim(h^q, h_i^d)` to select the most useful step during retrieval training.
- In the main implementation, the thinking depth is set to `m = 8`, and experiments show clear gains up to moderate depths with degradation when the chain becomes too long.
- The method is designed as a continuous latent analogue of chain-of-thought, avoiding the need to generate long natural-language reasoning traces during retrieval.
- CoD alone gives limited gains, which motivates the paper's additional self-distillation objective to compress useful intermediate signals into the final embedding.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ji-2025-more-2502-12974]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ji-2025-more-2502-12974]].
