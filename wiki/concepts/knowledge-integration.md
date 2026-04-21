---
type: concept
title: Knowledge Integration
slug: knowledge-integration
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge integration, 知识整合]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Integration** (知识整合) — the process of combining evidence distributed across multiple pieces of information into a unified representation or decision relevant to a query or task.

## Key Points

- HippoRAG frames knowledge integration as the missing capability in standard passage-isolated RAG systems.
- The method integrates information at retrieval time by spreading probability across a KG neighborhood instead of repeatedly alternating retrieval and generation.
- The paper argues that real applications such as literature review, legal briefing, and medical diagnosis demand this cross-passage integration behavior.
- Larger gains on MuSiQue and especially 2WikiMultiHopQA support the claim that the method helps when latent associations matter.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guti-rrez-2024-hipporag-2405-14831]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guti-rrez-2024-hipporag-2405-14831]].
