---
type: entity
title: NL2Bash
slug: nl2bash
date: 2026-04-20
entity_type: benchmark
aliases: [NL2Bash, NL to Bash]
tags: []
---

## Description

NL2Bash is a benchmark for translating natural-language requests into Bash commands, used in [[shi-2022-natural-2204-11454]] as the paper's shell-command generation testbed. Because direct execution is difficult to sandbox, the paper evaluates it with character-level BLEU and approximates semantic similarity with bashlex-based parsing.

## Key Contributions

- Provides the Bash-language setting in the paper's three-benchmark evaluation suite.
- Exposes the limits of execution-based selection when true command execution is unsafe or unavailable.

## Related Concepts

- [[code-generation]]
- [[executability-checking]]
- [[execution-based-decoding]]

## Sources

- [[shi-2022-natural-2204-11454]]
