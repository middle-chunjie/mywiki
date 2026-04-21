---
type: entity
title: Lean
slug: lean
date: 2026-04-20
entity_type: tool
aliases: [Lean theorem prover, Lean 4]
tags: []
---

## Description

Lean is the interactive theorem prover used throughout [[unknown-nd-leanstar]] for formal proof execution and verification. It provides the proof-state transition function, tactic legality checks, and the final success signal for complete proofs.

## Key Contributions

- Serves as the symbolic verifier that filters sampled trajectories into successful and unsuccessful proofs.
- Supplies the formal proof language and tactic interface that Lean-STaR learns to model.
- Makes it possible to use correctness-checked expert iteration rather than relying only on heuristic rewards.

## Related Concepts

- [[interactive-theorem-proving]]
- [[proof-state]]
- [[formal-theorem-proving]]

## Sources

- [[unknown-nd-leanstar]]
