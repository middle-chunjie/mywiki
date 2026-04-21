---
type: concept
title: Line-Diff Encoding
slug: line-diff-encoding
date: 2026-04-20
updated: 2026-04-20
aliases: [line diff representation, 行差分编码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Line-Diff Encoding** (行差分编码) — a representation of edits that marks each source line with change-status tags and placeholder anchors so insertions and deletions can be generated as structured line-level deltas.

## Key Points

- [[unknown-nd-coeditorleveraging-2305-18584]] encodes input code with status tokens `empty`, `<add>`, or `<del>` and placeholder tokens such as `<1> ... <n>` for the editable region.
- Output edits are emitted as placeholder-indexed insertion and deletion bundles, `EncOutput(Delta u) = <1>I_a D_a <2>I_{a+1} D_{a+1} ...`, which lets the decoder predict only changed content instead of copying the whole region.
- The paper argues this representation is shorter than alternatives based on direct post-edit generation or token-level edit tagging, making it a better fit for seq2seq transformers.
- The encoding explicitly forbids deleting a line that was already marked `<add>` in the input, constraining invalid or looping edit sequences.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-coeditorleveraging-2305-18584]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-coeditorleveraging-2305-18584]].
