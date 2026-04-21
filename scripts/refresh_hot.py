#!/usr/bin/env python3
"""Rebuild wiki/hot.md from wiki/log.md (last 7 days) + wiki/QUESTIONS.md (open) + existing hot.md's Current Focus.

Idempotent. Run on session Stop hook, or manually.

Output layout (## sections, in this order):
    ## Recent Additions        — last 7 days of ingest/query/reflect/merge/project log entries
    ## Open Questions          — lines under `## Open Questions` in wiki/QUESTIONS.md
    ## Active Projects         — non-archived projects from projects/*/PROJECT.md (omitted if none)
    ## Current Focus           — preserved verbatim from prior hot.md if present, else "_not set_"

Frontmatter: type: hot-cache, graph-excluded: true, date: today.
Body target: ≤500 words. If overflowing, oldest Recent Additions entries are trimmed first.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI = REPO_ROOT / "wiki"
HOT = WIKI / "hot.md"
LOG = WIKI / "log.md"
QUESTIONS = WIKI / "QUESTIONS.md"
PROJECTS = REPO_ROOT / "projects"

WORD_BUDGET = 500
RECENT_DAYS = 7
INTERESTING_OPS = {"ingest", "query", "reflect", "merge", "promote-notes", "create-project", "project-writeback", "archive-project"}


def _today() -> str:
    return dt.date.today().isoformat()


def _parse_log_line(line: str) -> tuple[dt.date | None, str, str]:
    # "YYYY-MM-DD HH:MM | op | subject"
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+\d{1,2}:\d{2}\s*\|\s*([^|]+?)\s*\|\s*(.*)$", line)
    if not m:
        return None, "", ""
    try:
        d = dt.date.fromisoformat(m.group(1))
    except ValueError:
        return None, "", ""
    return d, m.group(2).strip(), m.group(3).strip()


def recent_additions() -> list[str]:
    if not LOG.is_file():
        return []
    cutoff = dt.date.today() - dt.timedelta(days=RECENT_DAYS)
    entries: list[tuple[dt.date, str, str]] = []
    for raw in LOG.read_text(encoding="utf-8").splitlines():
        d, op, subj = _parse_log_line(raw)
        if d is None or d < cutoff:
            continue
        if op not in INTERESTING_OPS:
            continue
        entries.append((d, op, subj))
    # Newest first.
    entries.sort(key=lambda t: t[0], reverse=True)
    return [f"- `{d.isoformat()}` {op}: {subj}" for d, op, subj in entries]


def open_questions() -> list[str]:
    if not QUESTIONS.is_file():
        return []
    text = QUESTIONS.read_text(encoding="utf-8")
    m = re.search(r"^## Open Questions\s*$", text, re.MULTILINE)
    if not m:
        return []
    tail = text[m.end():]
    next_hdr = re.search(r"^## ", tail, re.MULTILINE)
    section = tail[: next_hdr.start()] if next_hdr else tail
    out: list[str] = []
    for ln in section.splitlines():
        s = ln.strip()
        if s.startswith("- [ ]"):
            out.append(s)
    return out


def active_projects() -> list[str]:
    if not PROJECTS.is_dir():
        return []
    try:
        import yaml  # type: ignore
    except ImportError:
        return ["_yaml not installed — cannot read project frontmatter_"]
    fm_re = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    entries: list[tuple[str, str, str]] = []
    for pdir in sorted(PROJECTS.iterdir()):
        pm = pdir / "PROJECT.md"
        if not pm.is_file():
            continue
        text = pm.read_text(encoding="utf-8")
        m = fm_re.match(text)
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception:
            continue
        stage = str(fm.get("stage", "unknown"))
        if stage == "archived":
            continue
        slug = fm.get("slug", pdir.name)
        title = fm.get("title", fm.get("direction", ""))
        updated = str(fm.get("updated", ""))
        entries.append((slug, stage, updated))
    return [f"- `{slug}` — stage: {stage}, updated: {updated}" for slug, stage, updated in entries]


def preserve_current_focus() -> str:
    if not HOT.is_file():
        return "_not set_"
    text = HOT.read_text(encoding="utf-8")
    m = re.search(r"^## Current Focus\s*$", text, re.MULTILINE)
    if not m:
        return "_not set_"
    tail = text[m.end():]
    next_hdr = re.search(r"^## ", tail, re.MULTILINE)
    focus = tail[: next_hdr.start()] if next_hdr else tail
    focus = focus.strip()
    return focus or "_not set_"


def trim_to_budget(sections: list[tuple[str, list[str] | str]]) -> list[tuple[str, list[str] | str]]:
    def word_count() -> int:
        total = 0
        for _, body in sections:
            if isinstance(body, list):
                total += sum(len(ln.split()) for ln in body)
            else:
                total += len(body.split())
        return total

    if word_count() <= WORD_BUDGET:
        return sections
    # Trim oldest Recent Additions entries first (they come last in the list since we sort newest-first).
    for title, body in sections:
        if title == "Recent Additions" and isinstance(body, list):
            while body and word_count() > WORD_BUDGET:
                body.pop()
            break
    return sections


def render(sections: list[tuple[str, list[str] | str]]) -> str:
    out: list[str] = []
    out.append("---")
    out.append("type: hot-cache")
    out.append("graph-excluded: true")
    out.append(f"date: {_today()}")
    out.append("---")
    out.append("")
    out.append("# Hot Cache")
    out.append("")
    out.append(
        "Session-bootstrap context. Regenerated by `scripts/refresh_hot.py`. "
        "Only the **Current Focus** section is user-editable; all others are overwritten."
    )
    out.append("")
    for title, body in sections:
        out.append(f"## {title}")
        out.append("")
        if isinstance(body, list):
            if body:
                out.extend(body)
            else:
                out.append("_none_")
        else:
            out.append(body)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    projects = active_projects()
    sections: list[tuple[str, list[str] | str]] = [
        ("Recent Additions", recent_additions()),
        ("Open Questions", open_questions()),
    ]
    if projects:
        sections.append(("Active Projects", projects))
    sections.append(("Current Focus", preserve_current_focus()))
    sections = trim_to_budget(sections)
    HOT.write_text(render(sections), encoding="utf-8")
    print(f"[refresh_hot] wrote {HOT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
