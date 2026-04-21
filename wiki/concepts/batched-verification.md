---
type: concept
title: Batched Verification
slug: batched-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [批量验证, speculative verification]
tags: [serving, speculation, retrieval, correctness]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Batched Verification** (批量验证) — a correctness-guarantee mechanism in speculative systems that groups multiple speculated results into a single batched ground-truth query, identifies the first mismatch, and rolls back computation only from that point.

## Key Points

- In the context of [[speculative-retrieval]], after `s` cache-based speculative retrievals with queries `{q_1, …, q_s}`, a single batched knowledge-base query retrieves `{d_1, …, d_s}`; mismatch index `m = argmin_i(d̂_i ≠ d_i)` determines the rollback point.
- Batched retrieval is strictly more efficient than `s` sequential queries due to I/O parallelism and amortized overhead, so the verification step costs less than the `s` individual retrievals it replaces.
- If `m = s + 1` (all speculations correct), no rollback occurs and the saving equals roughly `(s − 1)` retrieval calls; if `m = 1` (first step wrong), overhead equals one wasted speculation step.
- The expected number of verified documents per verification round is `(1 − γ^s) / (1 − γ)`, where `γ` is the per-step speculation accuracy.
- Asynchronous verification overlaps the batched query with the next speculation step, hiding verification latency when `b < a`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-accelerating-2401-14021]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-accelerating-2401-14021]].
