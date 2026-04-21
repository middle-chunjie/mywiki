---
type: entity
title: code-davinci-001
slug: code-davinci-001
date: 2026-04-20
entity_type: tool
aliases: [Codex code-davinci-001, code davinci 001]
tags: []
---

## Description

code-davinci-001 is the OpenAI Codex engine used as the primary completion model in [[shrivastava-2023-repositorylevel-2206-12839]].

## Key Contributions

- Serves as the black-box code model that RLPG prompts without access to model weights.
- Is queried with `temperature = 0.0`, newline stopping, completion length `24`, and maximum prompt length `4072`.

## Related Concepts

- [[code-language-model]]
- [[code-completion]]
- [[context-window]]

## Sources

- [[shrivastava-2023-repositorylevel-2206-12839]]
