---
type: concept
title: Image Geolocation
slug: image-geolocation
date: 2026-04-20
updated: 2026-04-20
aliases: [image geolocalization, visual geolocation, photo geolocation, 图像地理定位]
tags: [computer-vision, geospatial, information-retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Image Geolocation** (图像地理定位) — the task of predicting the geographic coordinates (latitude, longitude) of a photograph from its visual content alone, without any metadata or GPS signal.

## Key Points

- Two dominant paradigms: classification-based methods partition the Earth into discrete cells (e.g., Google S2 grid) and predict a cell label; retrieval-based methods compare a query image against a geotagged gallery and adopt the location of the nearest match.
- Classification methods introduce a hard accuracy ceiling tied to cell size; locations near cell boundaries incur systematic error.
- Retrieval methods are sensitive to gallery coverage and feature quality; a single nearest-neighbor match is brittle.
- Img2Loc reframes geolocation as text generation: an LMM is prompted with the query image plus GPS coordinates from retrieved positive and negative neighbors and asked to produce coordinates directly.
- Evaluation uses geodesic-distance thresholds: street (1 km), city (25 km), region (200 km), country (750 km), continent (2500 km).
- Img2Loc (GPT-4V) achieves `17.10 / 45.14 / 57.87 / 72.91 / 84.68` % on Im2GPS3k, surpassing the prior SOTA GeoCLIP at all granularities.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2024-imgloc-2403-19584]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2024-imgloc-2403-19584]].
