---
type: entity
title: Pascal VOC
slug: pascal-voc
date: 2026-04-20
entity_type: dataset
aliases: [PASCAL VOC, Pascal Visual Object Classes, Pascal-5i, Pascal-5^i]
tags: []
---

## Description

Pascal VOC (Visual Object Classes) is a benchmark dataset for object detection, segmentation, and classification, widely used as an evaluation benchmark in computer vision. Pascal-5^i is a derivative split with four non-overlapping subsets of five categories each, used for few-shot segmentation evaluation.

## Key Contributions

- Used in [[zhang-nd-what]] for both foreground segmentation (Pascal-5^i) and single object detection evaluations.
- The Pascal-5^i protocol provides four splits for cross-validated few-shot segmentation experiments; performance is averaged across all splits.
- Distribution shift experiments use Pascal as source domain and MS COCO as target domain.

## Related Concepts

- [[visual-in-context-learning]]
- [[few-shot-learning]]

## Sources

- [[zhang-nd-what]]
