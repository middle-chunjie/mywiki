---
type: concept
title: Deep Research
slug: deep-research
date: 2026-04-20
updated: 2026-04-20
aliases: [DR, 深度研究]
tags: [agents, retrieval, long-form]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Deep Research** (深度研究) — a tool-augmented language-model setting in which the system plans, searches, reads, and synthesizes multiple external sources into a long-form, attributed answer.

## Key Points

- The paper defines deep research as a language model equipped with search tools and explicit `think`, `tool`, `answer`, and `cite` actions.
- Unlike short-form QA, deep research requires long-form synthesis, source attribution, and judgment over multiple under-specified quality dimensions.
- DR Tulu targets science, healthcare, and general-domain deep-research tasks instead of proxying the setting with only verifiable factoid QA.
- The paper argues that realistic deep-research systems need adaptive evaluation because the relevant evidence changes as the policy explores the environment.
- Tool usage should be domain-adaptive: DR Tulu leans on paper search for scientific tasks and web search for broader open-domain tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shao-2025-dr-2511-19399]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shao-2025-dr-2511-19399]].
