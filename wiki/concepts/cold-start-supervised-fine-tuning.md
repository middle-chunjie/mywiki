---
type: concept
title: Cold-Start Supervised Fine-Tuning
slug: cold-start-supervised-fine-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [cold-start SFT, 冷启动监督微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cold-Start Supervised Fine-Tuning** (冷启动监督微调) — an initialization stage that teaches a base language-model agent how to interpret and apply retrieved skills before online reinforcement learning begins.

## Key Points

- SkillRL uses a teacher model to generate skill-augmented reasoning traces that demonstrate skill retrieval and skill-conditioned action choice.
- The policy is trained on this dataset with cross-entropy loss before any RL updates are applied.
- The resulting model becomes both the RL initialization and the KL reference policy for later GRPO optimization.
- Ablation shows this stage is critical: removing it drops performance from `89.9 / 72.7` to `65.2 / 46.5` on ALFWorld / WebShop.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2026-skillrl-2602-08234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2026-skillrl-2602-08234]].
