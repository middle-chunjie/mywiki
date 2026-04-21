---
type: entity
title: MetaICL
slug: metaicl
date: 2026-04-20
entity_type: model
aliases: [Meta ICL]
tags: []
---

## Description

MetaICL is a `774M` language model initialized from GPT-2 Large and meta-trained with an explicit in-context-learning objective. In this paper it is the clearest example of a model that relies heavily on demonstration format and only weakly on correct input-label mappings.

## Key Contributions

- Serves as the meta-trained baseline that shows only `0.1-0.9%` degradation from gold to random labels.
- Highlights how explicit meta-training can amplify dependence on simpler prompt cues such as format and label space.

## Related Concepts

- [[in-context-learning]]
- [[meta-training]]
- [[input-label-mapping]]

## Sources

- [[min-2022-rethinking-2202-12837]]
