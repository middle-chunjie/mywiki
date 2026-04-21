---
type: concept
title: Node Classification
slug: node-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [节点分类]
tags: [graph-learning, evaluation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Node Classification** (节点分类) — a graph learning task that predicts a label for each node from graph topology and node attributes.

## Key Points

- MA-GCL evaluates learned representations mainly through linear evaluation on node classification.
- The benchmarks span citation, co-author, and co-purchase graphs: Cora, CiteSeer, PubMed, Coauthor-CS, Amazon-Computers, and Amazon-Photo.
- On co-author and co-purchase graphs, the protocol uses `10%/10%/80%` train/validation/test splits.
- Reported performance is the mean and standard deviation over `5` random seeds.
- MA-GCL's empirical claim of superiority is primarily based on this task setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gong-2022-magcl-2212-07035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gong-2022-magcl-2212-07035]].
