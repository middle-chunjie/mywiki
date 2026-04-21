---
type: entity
title: InfoSeek-Eval
slug: infoseek-eval
date: 2026-04-20
entity_type: tool
aliases: [InfoSeek Eval, InfoSeekEval]
tags: []
---

## Description

InfoSeek-Eval is a 300-query in-domain evaluation split from the InfoSeek benchmark, used in [[zhou-2026-retrieve-2604-04949]] to assess task success rate and execution efficiency of LRAT-trained retrievers. The evaluation set is strictly disjoint from all LRAT training data.

## Key Contributions

- Serves as the primary in-domain benchmark for LRAT experiments, measuring Success Rate (SR) and Average Step Count.
- LRAT improvements are most pronounced here in terms of step reduction (up to ~30%), reflecting more efficient evidence acquisition.

## Related Concepts

- [[agentic-search]]
- [[deep-research-agent]]
- [[information-retrieval]]

## Sources

- [[zhou-2026-retrieve-2604-04949]]
