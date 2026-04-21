---
type: entity
title: MTP-CLNN
slug: mtp-clnn
date: 2026-04-20
entity_type: tool
aliases: [MTP-CLNN framework, Multi-task Pre-training with Contrastive Learning with Nearest Neighbors]
tags: [intent-discovery, contrastive-learning, dialogue, nlp]
---

## Description

MTP-CLNN is a two-stage framework for new intent discovery proposed by Zhang et al. (2022). Stage 1 (MTP) fine-tunes BERT with multi-task pre-training on external intent datasets and in-domain MLM. Stage 2 (CLNN) applies neighborhood-aware contrastive learning to produce compact, cluster-friendly utterance representations. Source code: https://github.com/zhang-yu-wei/MTP-CLNN.

## Key Contributions

- Achieves state-of-the-art performance on BANKING, StackOverflow, and M-CID in both unsupervised and semi-supervised NID as of 2022.
- Eliminates reliance on pseudo-labeling by using nearest-neighbor proximity as self-supervisory signal.
- Demonstrates strong label efficiency: performance degrades far less than prior methods when known-intent ratio is reduced.

## Related Concepts

- [[new-intent-discovery]]
- [[contrastive-learning-with-nearest-neighbors]]
- [[multi-task-learning]]
- [[bert]]

## Sources

- [[zhang-2022-new-2205-12914]]
