---
type: entity
title: ActivityNet
slug: activitynet
date: 2026-04-20
entity_type: dataset
aliases: [ActivityNet Captions]
tags: []
---

## Description

ActivityNet is the long-video benchmark used in [[jiang-2022-crossmodal-2211-09623]] for video-paragraph retrieval. The paper evaluates on the `val1` split and lengthens both caption and frame limits to `64`.

## Key Contributions

- Tests whether the adapter design remains effective for longer captions and longer videos than the default setting.
- Shows Cross-Modal Adapter reaches text-to-video `R@1 = 41.5`, essentially matching CLIP4Clip while training only `0.81M` parameters.

## Related Concepts

- [[text-video-retrieval]]
- [[query-aware-video-representation]]
- [[multi-modal-learning]]

## Sources

- [[jiang-2022-crossmodal-2211-09623]]
