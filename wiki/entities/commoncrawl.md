---
type: entity
title: CommonCrawl
slug: commoncrawl
date: 2026-04-20
entity_type: dataset
aliases: [Common Crawl]
tags: []
---

## Description

CommonCrawl is the open web-crawl repository that underlies many large pre-training corpora, including the URL-rich datasets used in [[gao-2025-metadata-2501-01956]]. The paper relies on CommonCrawl-style source URLs as cheap metadata for conditioning.

## Key Contributions

- Provides the raw URL metadata that makes MeCo practical without extra annotation cost.
- Motivates the use of absolute domains such as `en.wikipedia.org` as document-level grouping signals.
- Connects metadata conditioning to realistic web-scale pre-training pipelines rather than synthetic corpora only.

## Related Concepts

- [[metadata-conditioning]]
- [[language-model-pretraining]]
- [[pretraining-data-mixture]]

## Sources

- [[gao-2025-metadata-2501-01956]]
