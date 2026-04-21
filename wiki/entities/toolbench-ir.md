---
type: entity
title: ToolBench-IR
slug: toolbench-ir
date: 2026-04-20
entity_type: tool
aliases: [ToolBench IR]
tags: []
---

## Description

ToolBench-IR is a dense tool retriever used as a base model in [[unknown-nd-tool-2508-05152]]. TGR refines its tool embeddings with dependency-graph propagation and achieves the best retrieval results reported in the paper.

## Key Contributions

- Serves as the strongest text-embedding baseline in the main experiments.
- Provides the initial tool embeddings that TGR updates through graph convolution.
- Combined with TGR, reaches the reported SOTA results on API-Bank and ToolBench-I1.

## Related Concepts

- [[tool-retrieval]]
- [[dense-retrieval]]
- [[tool-embedding]]

## Sources

- [[unknown-nd-tool-2508-05152]]
