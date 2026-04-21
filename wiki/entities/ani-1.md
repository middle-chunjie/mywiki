---
type: entity
title: ANI-1
slug: ani-1
date: 2026-04-20
entity_type: dataset
aliases: [ANI-1 dataset]
tags: []
---

## Description

ANI-1 is a large off-equilibrium molecular dataset with roughly 20 million conformations, used in [[lu-2019-molecular]] to test whether MGCN transfers beyond equilibrium settings. In the paper it provides a harder large-scale benchmark than QM9.

## Key Contributions

- Tests MGCN on off-equilibrium molecules rather than only relaxed geometries.
- Shows MGCN reaching `MAE = 0.078`, outperforming SchNet and DTNN on the reported comparison.

## Related Concepts

- [[molecular-property-prediction]]
- [[transferability]]
- [[quantum-interaction]]

## Sources

- [[lu-2019-molecular]]
