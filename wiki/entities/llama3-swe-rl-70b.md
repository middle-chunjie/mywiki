---
type: entity
title: Llama3-SWE-RL-70B
slug: llama3-swe-rl-70b
date: 2026-04-20
entity_type: tool
aliases: [Llama 3 SWE RL 70B, Llama3-SWE-RL]
tags: []
---

## Description

Llama3-SWE-RL-70B is the reinforcement-learning-tuned model introduced in [[wei-2025-swerl-2502-18449]], built on top of Llama-3.3-70B-Instruct. It is trained on software-evolution-derived issue-solving data and evaluated with Agentless Mini.

## Key Contributions

- Achieves `41.0%` pass@1 on SWE-bench Verified in the paper's main setting.
- Outperforms the paper's SFT baseline on repair-only evaluation and several out-of-domain reasoning benchmarks.
- Demonstrates that RL on software issue solving can improve broader reasoning behavior.

## Related Concepts

- [[large-language-model]]
- [[reinforcement-learning]]
- [[domain-generalization]]

## Sources

- [[wei-2025-swerl-2502-18449]]
