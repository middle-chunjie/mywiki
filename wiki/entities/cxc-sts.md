---
type: entity
title: CxC-STS
slug: cxc-sts
date: 2026-04-20
entity_type: dataset
aliases: [Crisscrossed Captions STS, CxC]
tags: []
---

## Description

CxC-STS is the domain-shifted semantic textual similarity benchmark used in [[jiang-2022-improved-2203-06875]]. Its texts are image captions from the CxC extension of MS COCO, making it a harder transfer setting than standard STS.

## Key Contributions

- Tests robustness of sentence embeddings under domain shift without further adaptation.
- Uses sampled Spearman bootstrap correlation over `1000` resamples for evaluation.
- Shows PromCSE's largest reported gain over SimCSE: `71.2 ± 1.1` versus `67.5 ± 1.2` in the unsupervised setting.

## Related Concepts

- [[semantic-textual-similarity]]
- [[domain-shift]]
- [[sentence-embedding]]

## Sources

- [[jiang-2022-improved-2203-06875]]
