---
type: concept
title: Upper Confidence Bound
slug: upper-confidence-bound
date: 2026-04-20
updated: 2026-04-20
aliases: [UCB, UCT, Upper Confidence Bound applied to Trees, UCB1, upper confidence bound]
tags: [planning, search, exploration]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Upper Confidence Bound** (置信上界) — a family of selection strategies for sequential decision-making that balances exploitation of high-value options with exploration of less-visited ones by adding an optimistic uncertainty bonus to each candidate's estimated value.

## Key Points

- The basic UCT formula is `UCT_j = X̄_j + C * sqrt(2 * ln(N_C) / N_j)`, where `X̄_j` is the average reward, `N_C` the parent visit count, and `N_j` the child visit count; `C` tunes exploitation–exploration balance.
- In MCTSr ([[zhang-2024-accessing-2406-07394]]), UCT is adapted for LLM answer-refinement trees: `UCT_a = Q(a) + c * sqrt(ln(N(Father(a)) + 1) / (N(a) + eps))`, where `eps` avoids division by zero and the Q-value uses a custom min–mean blend.
- UCT was originally derived from the UCB1 bandit algorithm and applied to game-tree search (AlphaGo-style planning), enabling effective navigation of large branching factors.
- A node is eligible for selection only if it is not "fully expanded" — a condition requiring both a child count limit and at least one child exceeding the node's Q-value.
- The exploration constant `c` must be tuned per domain; in mathematical reasoning tasks the continuous, open-ended action space (infinite possible refinements) makes this calibration non-trivial.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-accessing-2406-07394]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-accessing-2406-07394]].
