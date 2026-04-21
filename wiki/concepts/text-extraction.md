---
type: concept
title: Text Extraction
slug: text-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [content extraction, boilerplate removal, 文本抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Text Extraction** (文本抽取) — the process of recovering the main natural-language content from raw documents such as HTML while removing surrounding boilerplate, navigation, and formatting artifacts.

## Key Points

- [[penedo-2023-refinedweb-2306-01116]] starts from raw CommonCrawl WARC files instead of `.WET` text because the latter retains too many menus, ads, and irrelevant fragments.
- The paper selects `trafilatura` as its extraction library after finding it cleaner than relying on preprocessed crawl text.
- After extraction, the pipeline normalizes formatting by limiting runs of newlines to `2` and removing literal URLs from the recovered text.
- The extracted text still requires later line-wise filtering, showing that extraction quality and downstream filtering are complementary rather than interchangeable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[penedo-2023-refinedweb-2306-01116]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[penedo-2023-refinedweb-2306-01116]].
