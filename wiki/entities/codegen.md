---
type: entity
title: CodeGen
slug: codegen
date: 2026-04-20
entity_type: model
aliases: [CodeGen-Multi]
tags: []
---

## Description

CodeGen is the family of Salesforce code language models evaluated in `350M`, `2B`, and `6B` parameter variants to test how MGD behaves across parameter scale.

## Key Contributions

- Shows that MGD improves all three scales on compilation, next-identifier match, identifier sequence match, and prefix match.
- Demonstrates that smaller monitored models can outperform larger unmonitored ones on some metrics.

## Related Concepts

- [[monitor-guided-decoding]]
- [[code-completion]]
- [[type-consistency]]

## Sources

- [[agrawal-nd-monitorguided]]
