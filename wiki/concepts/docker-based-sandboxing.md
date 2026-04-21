---
type: concept
title: Docker-Based Sandboxing
slug: docker-based-sandboxing
date: 2026-04-20
updated: 2026-04-20
aliases: [containerized execution sandbox, 基于 Docker 的沙箱化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Docker-Based Sandboxing** (基于 Docker 的沙箱化) — isolating executable tasks inside Docker containers so code can run reproducibly and safely without exposing the host environment.

## Key Points

- [[yang-2023-intercode-2306-14898]] uses Docker as the default execution substrate for all InterCode environments.
- A task designer specifies a Dockerfile, and the benchmark creates a stateful container that hosts the transition dynamics of the environment.
- Containerization enables reproducibility across machines and makes it easier to package system dependencies, databases, and file systems.
- The paper treats container isolation as essential for risky tasks such as filesystem-modifying Bash commands and future security-style challenges.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-intercode-2306-14898]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-intercode-2306-14898]].
