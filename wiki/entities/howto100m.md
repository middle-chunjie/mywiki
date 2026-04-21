---
type: entity
title: HowTo100M
slug: howto100m
date: 2026-04-20
entity_type: dataset
aliases: [HowTo100M dataset, HowTo100m]
tags: [dataset, video, instructional, pretraining, large-scale]
---

## Description

HowTo100M is a large-scale dataset of 136 million video clips from 1.2 million instructional YouTube videos with automatically transcribed narrations, introduced by Miech et al. (2019); used both as a pretraining corpus and as a source of pre-trained visual/text features.

## Key Contributions

- CrossCLR uses HowTo100M pre-trained features (provided by COOT) as visual expert input for Youcook2 experiments.
- The noisy video-narration pairs from HowTo100M motivated MIL-NCE's multiple positive formulation; CrossCLR builds on this by addressing false negatives more explicitly.

## Related Concepts

- [[multimodal-pretraining]]
- [[text-video-retrieval]]

## Sources

- [[zolfaghari-2021-crossclr-2109-14910]]
