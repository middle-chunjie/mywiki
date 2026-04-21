---
type: entity
title: FilM-7B
slug: film-7b
date: 2026-04-20
entity_type: tool
aliases: [FilM-7B, FILl-in-the-Middle]
tags: []
---

## Description

FilM-7B is the long-context model introduced in [[an-2024-make-2404-16811]], produced by applying [[information-intensive-training]] to [[mistral-7b-instruct-v0.2]].

## Key Contributions

- Reduces [[lost-in-the-middle]] behavior across document, code, and structured-data probing tasks.
- Improves average performance on nine real-world long-context tasks from `30.6` to `39.9` relative to its backbone.
- Maintains roughly unchanged short-context performance while improving long-context utilization.

## Related Concepts

- [[information-intensive-training]]
- [[lost-in-the-middle]]
- [[long-context-training]]

## Sources

- [[an-2024-make-2404-16811]]
