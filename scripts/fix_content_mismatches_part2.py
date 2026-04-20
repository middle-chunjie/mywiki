#!/usr/bin/env python3
"""Clean up the remaining 3 cases from Phase-1 mismatch fix:

1. csl_idx 614: huang-2024-effilearner-2405-19010 (wrong content) — delete entirely.
   The correct version huang-2024-effilearner-2405-15189 already exists.

2. csl_idx 234/238: the RRHF/SiTC pair got scrambled by the previous fix's order dependence.
   Currently:
     yuan-2023-rrhf-2304-14732       (bib=RRHF   md=SiTC)
     xu-2024-searchinthechain-2304-05302 (bib=SiTC md=RRHF)
   Target:
     yuan-2023-rrhf-2304-05302       (bib=RRHF   md=RRHF)
     xu-2024-searchinthechain-2304-14732 (bib=SiTC md=SiTC)
   Approach: delete both current, re-scaffold from decisions.jsonl with correct arxiv ids.

3. csl_idx 523 MemoRAG: bib fixed but download failed. Retry arxiv download + DeepXiv.
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from batch_scaffold import scaffold, make_reader  # noqa

RAW = REPO / "raw" / "papers"


def main() -> None:
    decs_by_idx = {d["csl_idx"]: d for d in (json.loads(l) for l in open("/tmp/mywiki-batch/decisions.jsonl"))}

    # 1. Delete wrong-content EffiLearner folder (csl_idx 614)
    bad = RAW / "huang-2024-effilearner-2405-19010"
    if bad.exists():
        shutil.rmtree(bad)
        print(f"[del] {bad.name} (wrong-content; 15189 correct-content kept)")

    # 2. Nuke + re-scaffold 234/238
    for name in ("yuan-2023-rrhf-2304-14732", "xu-2024-searchinthechain-2304-05302"):
        p = RAW / name
        if p.exists():
            shutil.rmtree(p)
            print(f"[del] {p.name} (scrambled by previous fix)")

    reader = make_reader()
    for idx, correct_arxiv in [(234, "2304.05302"), (238, "2304.14732"), (523, "2409.05591")]:
        dec = decs_by_idx[idx].copy()
        dec["arxiv_id"] = correct_arxiv
        # Let scaffold derive slug naturally from csl_item + correct arxiv_id
        dec.pop("slug", None)
        slug, status, note = scaffold(dec, reader)
        print(f"[{status}] csl_idx={idx} {slug} — {note}")


if __name__ == "__main__":
    main()
