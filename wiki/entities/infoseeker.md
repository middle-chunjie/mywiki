---
type: entity
title: InfoSeeker
slug: infoseeker
date: 2026-04-20
entity_type: tool
aliases: [InfoSeeker-3B]
tags: []
---

## Description

InfoSeeker is the compact deep-research and agentic-search model trained on the InfoSeek dataset. Its workflow combines explicit reasoning, parallel query generation, retrieval summarization, supervised fine-tuning, and GRPO-based reinforcement learning.

## Key Contributions

- Reaches `16.5%` accuracy on BrowseComp-Plus with `8.24` search calls.
- Outperforms several stronger open and closed baselines on search-intensive evaluation.
- Demonstrates that InfoSeek can transfer deep-research behavior into a `3B` model.

## Related Concepts

- [[agentic-search]]
- [[multi-query-search]]
- [[group-relative-policy-optimization]]

## Sources

- [[xia-2025-open-2509-00375]]
