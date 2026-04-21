---
type: entity
title: VGG-16
slug: vgg-16
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

VGG-16 is the CNN backbone used in [[sener-2018-active-1708-00489]] for all reported experiments and for constructing the feature space used by subset selection.

## Key Contributions

- Supplies the image representations whose final fully connected activations define the selection distance `Δ`.
- Serves as the supervised classifier retrained after each active-learning round.

## Related Concepts

- [[convolutional-neural-network]]
- [[active-learning]]
- [[core-set-selection]]

## Sources

- [[sener-2018-active-1708-00489]]
