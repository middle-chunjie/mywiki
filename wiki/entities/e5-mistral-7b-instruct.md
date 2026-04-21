---
type: entity
title: E5-Mistral-7B-Instruct
slug: e5-mistral-7b-instruct
date: 2026-04-20
entity_type: tool
aliases:
  - E5-mistral-7b-instruct
  - e5-mistral-7b-instruct
tags: []
---

## Description

E5-Mistral-7B-Instruct is the teacher retrieval model used in [[lee-2024-nvembed-2405-17428]] for positive-aware hard-negative mining.

## Key Contributions

- Scores candidate negatives during retrieval data curation so that likely false negatives can be filtered relative to the positive passage score.
- Serves as a strong disclosed decoder-only embedding baseline discussed throughout the paper.

## Related Concepts

- [[hard-negative-mining]]
- [[contrastive-learning]]
- [[dense-retrieval]]

## Sources

- [[lee-2024-nvembed-2405-17428]]
