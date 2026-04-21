---
type: entity
title: Amazon ESCI Dataset
slug: amazon-esci
date: 2026-04-20
entity_type: dataset
aliases: ["Amazon ESCI", "Shopping Queries Dataset", "KDD Cup 2022 ESCI"]
tags: [e-commerce, retrieval, benchmark]
---

## Description

Amazon ESCI (Exact, Substitute, Complement, Irrelevant) is a large-scale e-commerce search dataset released by Amazon for KDD Cup 2022, containing (query, item, relevance-label) triplets with four label types. It includes 68K unique queries and 1.2M items but lacks user identifiers, preventing personalization evaluation.

## Key Contributions

- Provides a public multi-functional benchmark for keyword search, query-by-example, and complementary item recommendation evaluation via its four label types.
- Used in the UIA paper as a public replication dataset alongside the private Lowe's dataset.
- Enables evaluation of joint information access modeling without personalization.

## Related Concepts

- [[unified-information-access]]
- [[query-by-example]]
- [[complementary-item-recommendation]]
- [[dense-retrieval]]

## Sources

- [[zeng-2023-personalized-2304-13654]]
