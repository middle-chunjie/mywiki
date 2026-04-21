---
type: entity
title: RepoCoder
slug: repocoder
date: 2026-04-20
entity_type: tool
aliases: [RepoCoder]
tags: []
---

## Description

RepoCoder is an iterative retrieval-generation framework for repository-level code completion, proposed by Zhang et al. (2023). It alternates between similarity-based retrieval of repository code snippets and frozen code language model generation, using each generation output as an augmented query for the next retrieval step. Source code is available at https://github.com/microsoft/CodeT/tree/main/RepoCoder.

## Key Contributions

- Introduced the iterative retrieval-generation loop that bridges the gap between an incomplete-code query and the intended completion target.
- Introduced the RepoEval benchmark with line, API invocation, and function body completion tasks.
- Demonstrated >10% EM improvement over in-file completion baselines across four code LMs.

## Related Concepts

- [[repository-level-code-completion]]
- [[retrieval-augmented-generation]]
- [[iterative-retrieval]]
- [[code-completion]]

## Sources

- [[zhang-2023-repocoder-2303-12570]]
