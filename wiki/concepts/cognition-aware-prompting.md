---
type: concept
title: Cognition-Aware Prompting
slug: cognition-aware-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [Three-Step Prompting, 认知感知提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cognition-Aware Prompting** (认知感知提示) — a prompting strategy that structures generation into staged cognitive operations so an LLM first understands context, then expands associations, and only then produces final outputs.

## Key Points

- The paper defines three stages: `Comprehension Synthesis`, `Associative Expansion`, and `Conclusion`.
- Step 1 extracts themes and intents to reduce false positives that drift away from the original search goal.
- Step 2 encourages novel but related expressions so the model expands variation without arbitrary hallucination.
- Step 3 conditions final generation on the earlier reasoning traces to produce cleaner paraphrases, intent shifts, and other augmentations.
- Replacing this prompting scheme with a naive template lowers QReCC MRR from `52.7` to `51.1`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-generalizing-2402-07092]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-generalizing-2402-07092]].
