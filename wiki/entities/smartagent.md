---
type: entity
title: SMARTAgent
slug: smartagent
date: 2026-04-20
entity_type: tool
aliases: [SMART Agent]
tags: []
---

## Description

SMARTAgent is the family of fine-tuned agent models introduced in [[qian-2025-smart-2502-11435]]. It is trained on SMART-ER to decide when to continue reasoning internally and when to invoke external tools.

## Key Contributions

- Reduces average tool usage by `24%` while improving macro performance by `37.1%`.
- Enables `7B`-scale models to approach or surpass much larger baselines in some domains.
- Generalizes to OOD evaluation with substantially fewer tool calls.

## Related Concepts

- [[tool-overuse]]
- [[interactive-inference]]
- [[confidence-calibration]]

## Sources

- [[qian-2025-smart-2502-11435]]
