---
type: entity
title: RetDec
slug: retdec
date: 2026-04-20
entity_type: tool
aliases: [RetDec]
tags: []
---

## Description

RetDec is the decompiler used in [[gui-2022-crosslanguage-2201-07420]] to convert non-obfuscated binary object files into LLVM-IR. It provides the binary-side IR input required by XLIR.

## Key Contributions

- Translates binaries into LLVM-IR so they can be embedded in the same representation space as source code.
- Enables the paper's cross-modal matching pipeline from binary artifacts to source snippets.
- The paper notes that severe obfuscation remains a limitation for this decompilation-based setup.

## Related Concepts

- [[llvm-ir]]
- [[intermediate-representation]]
- [[binary-source-code-matching]]

## Sources

- [[gui-2022-crosslanguage-2201-07420]]
