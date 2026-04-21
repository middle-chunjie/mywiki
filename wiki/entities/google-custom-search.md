---
type: entity
title: Google Custom Search
slug: google-custom-search
date: 2026-04-20
entity_type: tool
aliases: [GCS, Google Programmable Search Engine]
tags: []
---

## Description

Google Custom Search is the web retrieval API used in [[kasai-2024-realtime-2207-13332]] to gather current news documents for REALTIME QA. It serves as the freshest retrieval source among the paper's baseline configurations.

## Key Contributions

- Supplies top-ranked news articles that let open-book baselines answer newly changing questions.
- Enables the strongest baseline in the paper when combined with GPT-3 prompting.
- Highlights retrieval freshness as a larger bottleneck than reader capacity.

## Related Concepts

- [[up-to-date-information-retrieval]]
- [[open-domain-question-answering]]
- [[real-time-evaluation]]

## Sources

- [[kasai-2024-realtime-2207-13332]]
