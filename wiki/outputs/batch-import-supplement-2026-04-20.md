---
type: batch-import-supplement-report
graph-excluded: true
date: 2026-04-20
original_report: batch-import-2026-04-19.md
---

# Batch-import supplement report — 2026-04-20

Follow-up to `wiki/outputs/batch-import-2026-04-19.md`. Recovered arxiv_ids from title-only CSL entries (Zotero export had 331 papers where the CSL title was a PDF filename or the CSL lacked an arxiv mirror field), then re-ran the pipeline for the resolvable subset.

## Top-line numbers

| Stage | Value |
|---|---|
| Title-resolver candidates | 331 (222 uncertain + 109 partial-no-md) |
| Resolver hits (arxiv_id found) | 199 (60%) |
| Resolver misses | 132 (not on arxiv or too-garbled title) |
| Partial-no-md re-scaffolded | 66 / 67 resolved (1 PDF 404) |
| Uncertain re-classified → include | 59 / 132 resolved |
| Uncertain re-classified → still uncertain | 73 / 132 resolved |
| **Net new papers materialized** | **125** (66 partial + 59 uncertain) |

## Net effect on original report

Original counts (from `batch-import-2026-04-19.md`):
- include: 542 — of which 66 were `partial-no-md` now upgraded to `ok`
- uncertain: 222 — 59 now promoted to include, 73 remain uncertain, 90 unresolvable
- excluded: 47

Updated effective counts after this supplement:
- **ok (PDF + MD)**: 424 + 66 + 59 = **549**
- **partial-no-md** (waiting on manual PDF): 109 − 66 = **43**
- **failed** (PDF download 404): 9 + 1 = **10**
- **still uncertain** (user review recommended): 222 − 59 = **163** (73 re-classified undecided + 90 unresolvable-by-title)
- **excluded**: 47

## Detail reports

Two companion reports cover the two sub-phases:

- `batch-import-supplement-partial-2026-04-20.md` — 67 partial-no-md scaffolds (66 ok + 1 failed)
- `batch-import-supplement-uncertain-2026-04-20.md` — 59 uncertain reclassified as include

## Remaining uncertain worth flagging

The re-classifier flagged a handful of borderline cases where the enriched metadata just barely missed the threshold:

| arxiv_id | title | why close |
|---|---|---|
| 2310.00785 | BooookScore | ICLR 2024 (arxiv comment), but `resolved_venue` empty; cites=167 |
| 2305.18584 | CoEditor | ICLR 2024 (arxiv comment), but `resolved_venue` empty |
| 2303.03004 | xCodeEval | DBLP says ACL 2024 but `best_is_published=false` |
| 2409.04701 | Late Chunking | cites=48, one short of the 50+github threshold |
| 2502.06703 | s1: TTS Scaling 1B→405B | cites=128, no github_link |
| 2502.14768 | Logic-RL | cites=181, one short of 200 |
| 2505.04588 | ZeroSearch | 2025-05, one month before the Rule-2 cutoff; cites=115 |

If you want any of these in, run:
```bash
/batch-import <csl-json> --force-include "2310.00785,2305.18584,2303.03004,2409.04701,2502.06703,2502.14768,2505.04588"
```

## Remaining 132 unresolved titles

The resolver could not find an arxiv match for these (CSL titles too garbled, or the paper genuinely has no arxiv version — many are ACM/IEEE-only, workshop-only, or tech reports like AlphaCode 2). Manual triage options:

1. **If the title is recognizable**: manually look up the arxiv_id and run `/save-paper <arxiv_id>` one at a time.
2. **If it's ACM/IEEE-only**: drop `paper.pdf` into `raw/papers/<slug>/` manually, then `python scripts/mineru_ingest.py <slug>`.
3. **If you don't recognize it**: probably safe to ignore; can always re-import later if it becomes relevant.

The unresolved list is in `/tmp/mywiki-batch/resolved.jsonl` — items where `arxiv_id` is null.

## Pipeline artifacts (for re-runs)

- `/tmp/mywiki-batch/resolve_targets.jsonl` — 331 items fed to resolver
- `/tmp/mywiki-batch/resolved.jsonl` — resolver output (one line per target, arxiv_id or null)
- `/tmp/mywiki-batch/supplement_partial_decisions.jsonl` — 67 patched partial includes
- `/tmp/mywiki-batch/supplement_uncertain_enriched.jsonl` — 132 re-enriched uncertain items
- `/tmp/mywiki-batch/supplement_decisions/batch-*.json` — Sonnet classifier outputs
- `/tmp/mywiki-batch/supplement_uncertain_final_decisions.jsonl` — final merged 132 uncertain decisions (59 inc / 73 unc / 0 exc)
- `/tmp/mywiki-batch/supplement_uncertain_include_decisions.jsonl` — 59 scaffolded includes

## Next step

Run `/batch-ingest --auto` to ingest all 549 scaffolded papers (including the 125 new ones from this supplement) into `wiki/sources/`.

After ingest, you can review the 163 remaining uncertain items and optionally force-include the ones you want.
