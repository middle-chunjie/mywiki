---
type: concept
title: Search Agent
slug: search-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [web search agent, LLM-based search agent, search agent system]
tags: [information-retrieval, agent, large-language-model]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Search Agent** (搜索智能体) — an LLM-based system that autonomously orchestrates web browsing, query issuance, document reading, and answer synthesis into a coherent retrieval-and-reasoning pipeline, mimicking human-like interactive search behavior.

## Key Points

- Static search agents decompose the web-search process into a fixed pipeline of subtasks (query generation, document reading, knowledge synthesis), each handled by a specialized LLM module (e.g., WebGLM, LaMDA, SeeKeR).
- Dynamic agents (e.g., WebGPT) use reinforcement learning to train LLMs to freely issue search queries, scroll through results, and quote references, determined by the model itself rather than a pre-defined schedule.
- WebGPT is the pioneering dynamic agent: GPT-3 fine-tuned via RL with human feedback to use browser actions (query, click, quote) inside a simulated search environment.
- Search agents introduce new challenges around trustworthiness (factual grounding), bias propagation from low-quality web content, and user privacy.
- The key open research question is whether autonomous LLM search agents will supplant traditional index-based IR entirely or function as a complementary high-latency layer.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2024-large-2308-07107]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2024-large-2308-07107]].
