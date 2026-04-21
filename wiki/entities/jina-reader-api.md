---
type: entity
title: Jina Reader API
slug: jina-reader-api
date: 2026-04-20
entity_type: tool
aliases: [Jina Reader]
tags: []
---

## Description

Jina Reader API is the page-reading tool used in Search-o1 to fetch readable web-page content from retrieved URLs. It bridges the search results returned by Bing and the downstream Reason-in-Documents analysis step.

## Key Contributions

- Converts retrieved URLs into full page text for document reasoning.
- Provides the document content that Search-o1 compresses into refined evidence.

## Related Concepts

- [[reason-in-documents]]
- [[web-search]]
- [[retrieval-augmented-generation]]

## Sources

- [[li-2025-searcho-2501-05366]]
