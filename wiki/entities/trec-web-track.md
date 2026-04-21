---
type: entity
title: TREC Web Track
slug: trec-web-track
date: 2026-04-20
entity_type: benchmark
aliases: [TREC Web Track ad hoc task, Web Track]
tags: []
---

## Description

TREC Web Track is the evaluation benchmark used in [[qiao-2019-understanding-1904-07531]] to test BERT on classical ad hoc document ranking. It provides judged web queries that contrast with the paper's QA-oriented MS MARCO setting.

## Key Contributions

- Supplies `200` judged queries from the 2009-2012 Web Track tasks for the paper's ClueWeb experiments.
- Defines the official `NDCG@20` and `ERR@20` metrics used to assess whether BERT transfers from passage reranking to ad hoc retrieval.

## Related Concepts

- [[ad-hoc-retrieval]]
- [[information-retrieval]]
- [[reranking]]

## Sources

- [[qiao-2019-understanding-1904-07531]]
