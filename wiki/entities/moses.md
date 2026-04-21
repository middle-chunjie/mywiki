---
type: entity
title: MOSES
slug: moses
date: 2026-04-20
entity_type: benchmark
aliases: [Molecular Sets]
tags: []
---

## Description

MOSES is the molecular generative-model benchmark used in [[friedman-2023-vendi-2210-02410]] for evaluating diversity on generated molecules. The paper draws samples from benchmark models to compare Vendi Score against IntDiv.

## Key Contributions

- Supplies the molecular generation models used in the paper's diversity comparison.
- Provides a setting where [[vendi-score]] exposes duplicate-heavy outputs missed by average-pairwise metrics.

## Related Concepts

- [[vendi-score]]
- [[diversity-metric]]
- [[similarity-function]]

## Sources

- [[friedman-2023-vendi-2210-02410]]
