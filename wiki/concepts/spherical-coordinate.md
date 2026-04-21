---
type: concept
title: Spherical Coordinate
slug: spherical-coordinate
date: 2026-04-20
updated: 2026-04-20
aliases: [球坐标, spherical coordinates]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Spherical Coordinate** (球坐标) — a coordinate system that represents a vector by one radius and a sequence of angles rather than by Cartesian components.

## Key Points

- The paper uses spherical coordinates to reparameterize unit-norm embeddings from `d` Cartesian values into `d - 1` angles.
- In high dimensions, most polar angles of normalized vectors concentrate near `π/2`, which sharply reduces exponent entropy in float32 storage.
- The method computes `θ_i = arccos(x_i / ||x_{i:d}||_2)` for the first `d - 2` angles and `atan2` for the final angle to preserve quadrant information.
- The spherical form is also used for direct similarity computation through a backward recurrence, avoiding full Cartesian reconstruction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-00079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-00079]].
