---
type: entity
title: Gurobi
slug: gurobi
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

Gurobi is the optimization framework used in [[sener-2018-active-1708-00489]] to check feasibility of the robust `k`-center mixed-integer program.

## Key Contributions

- Solves the MIP subproblem used to refine the greedy `k`-center solution.
- Enables binary search over the feasible covering radius `δ`.

## Related Concepts

- [[k-center-problem]]
- [[mixed-integer-programming]]

## Sources

- [[sener-2018-active-1708-00489]]
