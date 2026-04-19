---
type: batch-import-supplement-v2-summary
graph-excluded: true
date: 2026-04-20
previous_reports:
  - batch-import-2026-04-19.md
  - batch-import-supplement-2026-04-20.md
---

# Batch-import v2 supplement — 2026-04-20

Follow-up to `batch-import-supplement-2026-04-20.md`. Two threads of work:

1. **Auto-fetch PDFs for non-arxiv papers** (42 remaining partial-no-md after v1)
2. **Re-classify 73 new-uncertain with broader rules** (add ACM TOIS/TPAMI/TKDE/PR/NN/etc. + known-enterprise tech-report whitelist)

## Top-line numbers

| Action | Count |
|---|---|
| Non-arxiv PDFs auto-fetched (Unpaywall/OpenAlex/S2) | **15 / 42** (~36%) |
| V2 Sonnet includes (with broader rules) | **12 / 73** |
| New paper folders materialized | **12** (v2 includes are all new) |
| PDFs still needing mineru | 15 (fetched via Unpaywall, needs paper.md) |
| Still uncertain after v2 | 60 |
| Excluded in v2 (physics out-of-domain) | 1 |

## Pipeline: non-arxiv PDF fetcher

New script: `scripts/fetch_nonarxiv_pdf.py`. Tries 4 sources in order:
1. **Direct URL** (if CSL URL ends in `.pdf` or is `aclanthology.org` / `openreview.net`)
2. **Unpaywall** (requires only email; best-quality signal)
3. **OpenAlex** (by DOI, fallback by title)
4. **Semantic Scholar** (by DOI, fallback by title)

**Hit rate on 42 partial-no-md**: 15 OK (ACL Anthology + few Springer/Elsevier OA) + 27 FAIL (paywalled: ACM DL, IEEE Xplore). For the paywalled 27, manual download is still required.

Usage:
```bash
# Single paper
python scripts/fetch_nonarxiv_pdf.py --doi "10.18653/v1/2022.acl-long.353" --out /tmp/p.pdf

# Batch from decisions JSONL
python scripts/fetch_nonarxiv_pdf.py --decisions <jsonl> --out-dir raw/papers --report report.json
```

## Rule changes (updated in `.claude/skills/batch-import/SKILL.md`)

**Equivalent-tier whitelist expanded** — these now qualify as "CCF-B-or-above":
- ML/AI: TMLR, MLSys, CoLT, COLM, ICLR, JMLR, AISTATS, UAI, Nature/Science/PNAS
- IR/DM: **ACM TOIS**
- Software eng: **ACM TOSEM**
- Vision: **IEEE TPAMI, Pattern Recognition (Elsevier)**
- Data/KE: **IEEE TKDE**
- Neural computation: **Neural Networks (Elsevier)**
- Surveys: **ACM Computing Surveys**
- HCI/PL: CHI, UIST, OOPSLA, POPL, PLDI

**New rule: enterprise technical reports** — include even without venue if from DeepMind / Google / OpenAI / Anthropic / Meta / FAIR / Microsoft / Apple / IBM Research / NVIDIA Research. Detection via author affiliation, byline, or title signals ("Technical Report", "System Card", product names like Gemini/Claude/GPT-X/Llama-X/AlphaCode).

## V2 re-classification — 12 newly promoted to include

| arxiv_id | title | flip reason |
|---|---|---|
| 2310.00785 | BooookScore | arxiv_comment → ICLR 2024 camera-ready |
| 2305.18584 | CoEditor | arxiv_comment → ICLR 2024 |
| 2312.11805 | Gemini (Google tech report) | Rule 2 — enterprise whitelist |
| 2311.11829 | System 2 Attention | Rule 2 — Meta/FAIR authors |
| 2403.20327 | Gecko | Rule 2 — Google Research |
| 2303.03004 | xCodeEval | DBLP secondary hit → ACL 2024 |
| 2301.09043 | CodeScore | ACM TOSEM (new tier) |
| 2310.09748 | LAIL (code ICL) | ACM TOSEM |
| 2211.09623 | Cross-Modal Adapter | Pattern Recognition (new tier) |
| 2312.15234 | LLM Serving Survey | ACM Computing Surveys (new tier) |
| 2307.08303 | SPTAR | Knowledge-Based Systems (CCF-B) |
| 2401.12954 | Meta-Prompting | cites=129 + github in arxiv_comment |

All 12 scaffolded successfully (arxiv PDF + DeepXiv paper.md). Report: `batch-import-supplement-v2-2026-04-20.md`.

## Cumulative state of migration

Original 2026-04-19 counts → after v1 supplement (2026-04-20) → after v2 supplement:

| Bucket | v0 | v1 | v2 |
|---|---|---|---|
| `ok` (PDF + MD) | 424 | 549 | **561** |
| `partial-no-md` with PDF but no MD (mineru queue) | 0 | 43 | **43 + 15 = 58** |
| `failed` (PDF download error) | 9 | 10 | 10 |
| Remaining `uncertain` (resolver missed + Sonnet kept) | 222 | 163 | **151** |
| `excluded` | 47 | 47 | 48 |

**Current `raw/papers/` count: 592 folders.**

## Remaining manual work

1. **Run mineru on 15 newly-fetched non-arxiv PDFs** to produce `paper.md`:
   ```bash
   for slug in $(cat /tmp/mywiki-batch/v2_need_mineru.txt); do
       python scripts/mineru_ingest.py "$slug"
   done
   ```
   Slow (~30s/paper, rate limits). Safe to run in background.

2. **27 paywalled papers** (ACM DL, IEEE Xplore) — need institutional proxy + manual PDF drop.

3. **151 still-uncertain** — v2 couldn't recover; user review recommended for any known-important papers.

## Next step

Once mineru finishes the 15 non-arxiv PDFs:
```bash
/batch-ingest --auto
```
… will ingest all 561+ scaffolded papers into `wiki/sources/`.
