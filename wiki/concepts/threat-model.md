---
type: concept
title: Threat Model
slug: threat-model
date: 2026-04-20
updated: 2026-04-20
aliases: [adversarial threat model, 威胁模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Threat Model** (威胁模型) — a formal specification of an attacker's capabilities, information, and constraints relative to a defender or monitoring system.

## Key Points

- The paper's MRT setup uses threat models to fix the informational asymmetry between agent and monitor before comparing methods.
- Its core design crosses agent awareness with monitor awareness to produce `6` distinct settings.
- The monitor is assumed to see full trajectories, while the red team has API access but not direct access to the victim model internals.
- The paper argues that conclusions about monitor reliability are not stable unless the threat model is stated explicitly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
