---
type: concept
title: Identifier Substitution Attack
slug: identifier-substitution-attack
date: 2026-04-20
updated: 2026-04-20
aliases: [identifier renaming attack, variable renaming adversarial attack, 标识符替换攻击]
tags: [adversarial-attack, source-code, software-engineering, robustness]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Identifier Substitution Attack** (标识符替换攻击) — a black-box adversarial attack on neural source-code models that renames programmer-defined identifiers (method names, variable names) with semantically similar alternatives, producing code that is syntactically valid, functionally equivalent, yet causes significant degradation in model output quality.

## Key Points

- Exploits the fact that programmer-defined identifiers do not affect program execution semantics but are heavily relied upon by neural code models trained on surface-form token sequences.
- Candidate replacements are selected using word2vec cosine similarity, ensuring the substituted identifier is semantically close to the original — this maintains the surface plausibility of the adversarial example.
- Identifier saliency `S(p, w) = cos(vec(w), vec(p))` quantifies how contextually important each identifier is to the program representation; combined with the score-change metric `Δscore_w*`, the joint ranking function `H(p, p*, w)` guides which identifiers to substitute first.
- The attack is fully black-box (only model outputs, not gradients, are required), making it applicable to deployed code analysis systems.
- Compilability is preserved by excluding identifiers already present in the program from the replacement candidate pool.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2022-adversarial]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2022-adversarial]].
