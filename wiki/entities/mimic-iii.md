---
type: entity
title: MIMIC-III
slug: mimic-iii
date: 2026-04-20
entity_type: dataset
aliases: [MIMIC III, Medical Information Mart for Intensive Care III]
tags: []
---

## Description

MIMIC-III is the clinical notes dataset used in [[morris-2023-text-2310-06816]] for a privacy case study. The paper evaluates a pseudo re-identified version in which fake names replace the original deidentified spans.

## Key Contributions

- Provides a sensitive-domain test showing that Vec2Text recovers `94.2%` of first names, `95.3%` of last names, and `89.2%` of full names.
- Demonstrates that even when exact recovery fails, the reconstructed notes often remain semantically close to the originals.

## Related Concepts

- [[privacy-leakage]]
- [[text-embedding]]
- [[embedding-inversion]]

## Sources

- [[morris-2023-text-2310-06816]]
