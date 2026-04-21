---
type: concept
title: Language Server Protocol
slug: language-server-protocol
date: 2026-04-20
updated: 2026-04-20
aliases: [LSP]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Language Server Protocol** (语言服务器协议) — an interface standard between development tools and language-specific analysis backends that exposes services such as completion, type resolution, and diagnostics.

## Key Points

- The paper uses LSP as the abstraction layer between monitors and programming-language-specific static analysis engines.
- This design lets the same monitoring interface generalize across Java, C#, Rust, and potentially other languages.
- Eclipse JDT.LS is the concrete Java engine used for the main type-consistent dereference monitor.
- The released `multilspy` library is presented as a language-agnostic client for interacting with language servers in AI-for-code settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[agrawal-nd-monitorguided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[agrawal-nd-monitorguided]].
