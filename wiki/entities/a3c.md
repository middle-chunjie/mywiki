---
type: entity
title: A3C
slug: a3c
date: 2026-04-20
entity_type: tool
aliases: [Asynchronous Advantage Actor-Critic]
tags: []
---

## Description

A3C is the reinforcement-learning algorithm used to pre-train the Sokoban policy network in [[orseau-2018-singleagent-1811-10928]].

## Key Contributions

- Supplies the fixed Markovian policy that guides LevinTS, LubyTS, and multiTS in the experiments.
- Enables the paper to test whether learned action priors can be converted into stronger search procedures.

## Related Concepts

- [[policy-guided-search]]
- [[markovian-policy]]
- [[single-agent-search]]

## Sources

- [[orseau-2018-singleagent-1811-10928]]
