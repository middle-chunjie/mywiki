---
type: entity
title: Im2GPS3k
slug: im2gps3k
date: 2026-04-20
entity_type: tool
aliases: [Im2GPS 3k, Im2GPS3K]
tags: []
---

## Description

Im2GPS3k is a benchmark dataset of 3,000 geotagged images used to evaluate worldwide image geolocation methods, introduced by Vo et al. (ICCV 2017). Performance is measured by the fraction of predictions falling within 1 km, 25 km, 200 km, 750 km, and 2500 km of the ground-truth coordinate.

## Key Contributions

- Serves as one of two primary evaluation datasets in the Img2Loc paper; Img2Loc (GPT-4V) achieves 17.10/45.14/57.87/72.91/84.68 % across the five distance thresholds.
- Establishes a standard multi-granularity evaluation protocol (street → city → region → country → continent) widely adopted in geolocation research.

## Related Concepts

- [[image-geolocation]]
- [[image-retrieval]]

## Sources

- [[zhou-2024-imgloc-2403-19584]]
