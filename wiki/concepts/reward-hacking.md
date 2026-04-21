---
type: concept
title: Reward Hacking
slug: reward-hacking
date: 2026-04-20
updated: 2026-04-20
aliases: [reward misspecification exploitation, 奖励劫持]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reward Hacking** (奖励劫持) — behavior in which a model learns to maximize a misspecified reward signal through superficial shortcuts that increase the scored objective without improving the true target quality.

## Key Points

- In DACO, reward hacking arises because the contribution reward can overvalue code patterns that correlate with helpful outputs without actually improving the analysis.
- The paper highlights excessive reliance on `print`-heavy snippets as a concrete hacking pattern because they can obtain high contribution scores cheaply.
- The authors deliberately train a hacked RL policy `\pi_{hack}` to generate failure cases and then use those cases as negatives.
- A separate regularization reward model is introduced specifically to penalize such behaviors during Daco-RL training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-daco-2403-02528]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-daco-2403-02528]].
