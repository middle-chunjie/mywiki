---
type: entity
title: DocT5Query
slug: doct5query
date: 2026-04-20
entity_type: model
aliases: [doc2query-T5]
tags: []
---

## Description

DocT5Query is a document-expansion retrieval method cited in [[gao-2021-coil-2104-07186]] as a language-model-enhanced lexical baseline. It augments documents with generated queries to improve lexical recall under vocabulary mismatch.

## Key Contributions

- Serves as a strong neural-lexical baseline that still trails COIL on the reported MS MARCO passage and document results.
- Exemplifies a retrieval strategy that improves lexical matching indirectly through document expansion rather than contextual token scoring.
- Helps clarify that COIL's gains do not come only from adding language-model information to a lexical index, but from changing the scoring signal itself.

## Related Concepts

- [[information-retrieval]]
- [[vocabulary-mismatch]]
- [[exact-lexical-match]]

## Sources

- [[gao-2021-coil-2104-07186]]
