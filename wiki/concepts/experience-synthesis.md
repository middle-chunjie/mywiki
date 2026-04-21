---
type: concept
title: Experience Synthesis
slug: experience-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic experience, synthesized rollouts]
tags: [agents, simulation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Experience Synthesis** (经验合成) — the generation of synthetic trajectories or transitions that expand training and verification coverage when direct interaction with the real environment is expensive or rate-limited.

## Key Points

- ASG-SI uses synthesized experience mainly for stress testing skill interfaces and broadening curriculum coverage.
- The paper explicitly treats synthetic data as complementary to real rollouts rather than a full replacement.
- Promotion criteria are meant to be re-grounded with real-environment replay so synthetic distribution shift does not dominate skill promotion.
- The authors highlight adversarial and rare-edge-case generation as a security-aligned use of synthesis.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2025-audited-2512-23760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2025-audited-2512-23760]].
