---
type: concept
title: Information Popularity
slug: information-popularity
date: 2026-04-20
updated: 2026-04-20
aliases: [popularity of information]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Information Popularity** (信息流行度) — the relative prevalence of an entity or fact in accessible corpora, often used as a proxy for how likely a model is to have memorized it.

## Key Points

- KITAB uses the number of Wikidata sitelinks for an author as a proxy for information popularity because direct training-frequency measurements are unavailable.
- The paper finds a sharp drop in irrelevant books once author popularity moves out of the lowest bins, but stronger popularity does not guarantee high completeness or correctness.
- Popularity is therefore treated as one explanatory factor for parametric recall, not as a sufficient explanation for successful constraint verification.
- The benchmark exposes different failure modes for low-popularity authors versus high-popularity authors under the same constraint types.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-kitabevaluating-2310-15511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-kitabevaluating-2310-15511]].
