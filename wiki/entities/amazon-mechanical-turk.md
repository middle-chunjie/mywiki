---
type: entity
title: Amazon Mechanical Turk
slug: amazon-mechanical-turk
date: 2026-04-20
entity_type: platform
aliases:
  - MTurk
tags: []
---

## Description

Amazon Mechanical Turk is the crowdsourcing platform used in [[samarinas-2024-procis]] to collect graded relevance labels and evidence spans for the benchmark's test set. The paper uses it to turn pooled candidate lists into a densely judged evaluation set.

## Key Contributions

- Hosts the `1,000` HITs used for ProCIS test annotation.
- Enables majority-vote relevance labels and highlighted evidence spans across `3` workers per HIT.

## Related Concepts

- [[relevance-judgment]]
- [[depth-k-pooling]]
- [[proactive-retrieval]]

## Sources

- [[samarinas-2024-procis]]
