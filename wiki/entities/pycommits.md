---
type: entity
title: PyCommits
slug: pycommits
date: 2026-04-20
entity_type: dataset
aliases: [PyCommits dataset]
tags: []
---

## Description

PyCommits is the commit-history dataset introduced in [[unknown-nd-coeditorleveraging-2305-18584]] for training and evaluating multi-round code auto-editing on Python repositories. It is built from `1,650` permissively licensed open-source projects mined from [[github]].

## Key Contributions

- Provides `1,550/50/50` train/valid/test project splits with commit-derived unit modifications as editing targets.
- Supplies synthetic repeated-editing examples by inlining a subset of changed lines into the model input.
- Anchors the paper's large-scale evaluation with `217K` training commits and `7.10M` modified training lines.

## Related Concepts

- [[code-editing]]
- [[multi-round-editing]]
- [[repository-level-context]]

## Sources

- [[unknown-nd-coeditorleveraging-2305-18584]]
