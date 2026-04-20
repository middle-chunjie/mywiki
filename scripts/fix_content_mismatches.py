#!/usr/bin/env python3
"""Fix the 6 Phase-1 Sonnet arxiv-id swaps.

For each mismatch:
  1. Compute correct_arxiv from CSL URL/note (authoritative = user's Zotero intent).
  2. Old folder is named with the WRONG arxiv suffix. Rename to the correct suffix.
  3. Delete the stale paper.pdf + paper.md (wrong content) inside the folder.
  4. Patch paper.bib's `note = {arXiv:<correct>}` and drop/rewrite stale URL.
  5. Re-download paper.pdf from arxiv + fetch paper.md via DeepXiv.

Uses scripts/batch_scaffold helpers (download_arxiv_pdf, fetch_deepxiv_md, make_reader).
"""
from __future__ import annotations
import json, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from batch_scaffold import download_arxiv_pdf, fetch_deepxiv_md, make_reader, make_slug  # noqa

RAW_PAPERS = REPO / "raw" / "papers"

# csl_idx -> correct arxiv_id (per CSL), plus the wrong arxiv currently in slug/bib.
FIXES = [
    {"csl_idx": 167, "wrong": "2306.03409", "correct": "2306.03438"},
    {"csl_idx": 234, "wrong": "2304.14732", "correct": "2304.05302"},
    {"csl_idx": 238, "wrong": "2304.05302", "correct": "2304.14732"},
    {"csl_idx": 424, "wrong": "2402.01030", "correct": "2406.00456"},
    {"csl_idx": 523, "wrong": "2409.09185", "correct": "2409.05591"},
    {"csl_idx": 614, "wrong": "2405.19010", "correct": "2405.15189"},
]

def _slug_suffix(arxiv_id: str) -> str:
    # make_slug uses slugify which replaces non-alnum with '-' — so 2405.19010 → 2405-19010.
    return "-" + re.sub(r"[^a-zA-Z0-9]+", "-", arxiv_id).strip("-")

def find_folder_with_suffix(wrong_arxiv: str) -> Path | None:
    suffix = _slug_suffix(wrong_arxiv)
    for p in RAW_PAPERS.iterdir():
        if p.is_dir() and p.name.endswith(suffix):
            return p
    return None

def rename_slug(old_name: str, wrong_arxiv: str, correct_arxiv: str) -> str:
    old_suffix = _slug_suffix(wrong_arxiv)
    new_suffix = _slug_suffix(correct_arxiv)
    if old_name.endswith(old_suffix):
        return old_name[: -len(old_suffix)] + new_suffix
    return old_name

def patch_bib(bib_text: str, correct_arxiv: str, wrong_arxiv: str) -> str:
    # Update note = {arXiv:XXX}
    bib = re.sub(
        r"note\s*=\s*\{\s*arXiv:[\d\.]+\s*\}",
        f"note = {{arXiv:{correct_arxiv}}}",
        bib_text,
    )
    # Replace wrong arxiv in URL / note / number fields if present
    bib = bib.replace(wrong_arxiv, correct_arxiv)
    return bib

def main():
    reader = make_reader()
    report = []
    for fix in FIXES:
        wrong, correct = fix["wrong"], fix["correct"]
        folder = find_folder_with_suffix(wrong)
        if folder is None:
            print(f"[skip] csl_idx={fix['csl_idx']}: no folder ending in {wrong}", file=sys.stderr)
            report.append({**fix, "status": "skip-nofolder"})
            continue

        new_name = rename_slug(folder.name, wrong, correct)
        new_folder = RAW_PAPERS / new_name
        if new_folder.exists() and new_folder != folder:
            print(f"[skip] csl_idx={fix['csl_idx']}: target folder {new_name} already exists", file=sys.stderr)
            report.append({**fix, "old": folder.name, "target": new_name, "status": "skip-target-exists"})
            continue

        # Delete stale content
        for fn in ("paper.pdf", "paper.md", "images"):
            p = folder / fn
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

        # Patch bib
        bib_path = folder / "paper.bib"
        if bib_path.exists():
            bib_path.write_text(patch_bib(bib_path.read_text(encoding="utf-8"), correct, wrong), encoding="utf-8")

        # Rename folder
        folder.rename(new_folder)
        folder = new_folder

        # Re-download
        pdf_ok = download_arxiv_pdf(correct, folder / "paper.pdf")
        md_ok = False
        if pdf_ok and reader is not None:
            md_ok = fetch_deepxiv_md(reader, correct, folder / "paper.md")

        status = "ok" if (pdf_ok and md_ok) else ("partial" if pdf_ok else "failed")
        print(f"[{status}] csl_idx={fix['csl_idx']} {folder.name}  (pdf={pdf_ok} md={md_ok})")
        report.append({**fix, "folder": folder.name, "pdf": pdf_ok, "md": md_ok, "status": status})

    with open("/tmp/mywiki-batch/fix_content_mismatches_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nreport: /tmp/mywiki-batch/fix_content_mismatches_report.json")

if __name__ == "__main__":
    main()
