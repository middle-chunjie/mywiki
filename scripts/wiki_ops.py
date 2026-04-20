#!/usr/bin/env python3
"""Unified bookkeeping CLI for the MyWiki knowledge base.

Offloads frontmatter arithmetic and markdown-append chores from Claude
skills to a single deterministic script. See wiki/outputs/optimization-plan-2026-04-20.md
for the design rationale.

Subcommands:
  cascade-update       strengthen / create concept pages after an INGEST
  index-update         manipulate wiki/index.md sections (paper, synthesis, resolve question)
  log-append           append a timestamped entry to wiki/log.md
  rejection-append     record a rejected draft in wiki/rejections.md
  merge-execute        merge concept/entity page <drop> into <keep> (dry-run by default)

All subcommands read/write files in-place. Errors exit non-zero with a
one-line stderr message.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    print("wiki_ops: PyYAML required (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
CONCEPTS_DIR = WIKI_DIR / "concepts"
SOURCES_DIR = WIKI_DIR / "sources"
INDEX_PATH = WIKI_DIR / "index.md"
LOG_PATH = WIKI_DIR / "log.md"
QUESTIONS_PATH = WIKI_DIR / "QUESTIONS.md"
REJECTIONS_PATH = WIKI_DIR / "rejections.md"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
SECTION_RE = re.compile(r"^(## .+)$", re.MULTILINE)


def _today() -> str:
    return dt.date.today().isoformat()


def _now_hhmm() -> str:
    return dt.datetime.now().strftime("%H:%M")


def _die(msg: str, code: int = 1) -> None:
    print(f"wiki_ops: {msg}", file=sys.stderr)
    sys.exit(code)


# ----------------------------- frontmatter helpers -----------------------------


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (fm_text, body). fm_text excludes the delimiters. body starts after the second '---\\n'."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return "", text
    return m.group(1), text[m.end():]


def _parse_fm(fm_text: str) -> dict[str, Any]:
    if not fm_text:
        return {}
    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        _die(f"YAML parse error: {e}")
    if not isinstance(data, dict):
        _die("frontmatter is not a mapping")
    return data


def _set_fm_field(fm_text: str, key: str, new_value: Any) -> str:
    """Patch a frontmatter line in-place. Preserves ordering/comments for untouched lines.

    If key doesn't exist, append it just before the closing delimiter's side of fm_text.
    """
    rendered = _render_fm_value(new_value)
    line_re = re.compile(rf"^({re.escape(key)}):[ \t]*(.*?)(\s*#.*)?$", re.MULTILINE)
    if line_re.search(fm_text):
        return line_re.sub(lambda m: f"{key}: {rendered}" + (m.group(3) or ""), fm_text, count=1)
    sep = "\n" if fm_text and not fm_text.endswith("\n") else ""
    return fm_text + sep + f"{key}: {rendered}\n"


def _render_fm_value(v: Any) -> str:
    def _dump(x: Any) -> str:
        # strip YAML end-of-document marker '...' and trailing whitespace
        return yaml.safe_dump(x, default_flow_style=True).rstrip().removesuffix("...").rstrip()
    if isinstance(v, list):
        return _dump(v)
    if isinstance(v, str):
        if re.search(r"[:#&*!,\[\]\{\}>|]", v) or v.strip() != v:
            return _dump(v)
        return v
    return _dump(v)


def _write_with_fm(path: Path, fm_text: str, body: str) -> None:
    # Always emit exactly one blank line between closing '---' and body, matching the template convention.
    body_stripped = body.lstrip("\n")
    out = f"---\n{fm_text.rstrip()}\n---\n\n{body_stripped}"
    path.write_text(out, encoding="utf-8")


# ----------------------------- body section helpers -----------------------------


def _find_section(body: str, title: str) -> tuple[int, int]:
    """Return (start, end) char offsets of the section's CONTENT (after the heading,
    before the next `## ` heading or EOF). (-1, -1) if missing."""
    header_pat = re.compile(rf"^## {re.escape(title)}\s*$", re.MULTILINE)
    m = header_pat.search(body)
    if not m:
        return -1, -1
    content_start = m.end()
    # Next ## heading
    next_m = re.search(r"^## ", body[content_start:], re.MULTILINE)
    content_end = content_start + next_m.start() if next_m else len(body)
    return content_start, content_end


def _section_text(body: str, title: str) -> str:
    s, e = _find_section(body, title)
    return "" if s < 0 else body[s:e]


def _append_to_section(body: str, title: str, line: str, dedupe: bool = False) -> tuple[str, bool]:
    """Append `line` (no trailing newline) inside the named section.

    Returns (new_body, appended). If dedupe=True and line already present (as full line), skip.
    """
    s, e = _find_section(body, title)
    if s < 0:
        # Append section at end of file.
        sep = "" if body.endswith("\n") else "\n"
        new = body + sep + f"\n## {title}\n\n{line}\n"
        return new, True

    section = body[s:e]
    if dedupe and re.search(rf"^{re.escape(line)}\s*$", section, re.MULTILINE):
        return body, False

    # Find where to insert: right before the section's trailing blank line(s) that precede next ##.
    # Simpler: trim trailing whitespace within the section, insert line, restore one blank.
    section_stripped = section.rstrip() + "\n"
    new_section = section_stripped + line + "\n"
    # Preserve trailing blank if original had one or if there's a next section
    if e < len(body):
        new_section += "\n"
    return body[:s] + new_section + body[e:], True


def _section_has_content(body: str, title: str) -> bool:
    """Section has 'real' content when, stripped of HTML comments and whitespace, it's non-empty."""
    txt = _section_text(body, title)
    stripped = HTML_COMMENT_RE.sub("", txt).strip()
    return bool(stripped)


# ----------------------------- confidence arithmetic -----------------------------


def _new_confidence(new_count: int, has_contradictions: bool, old_conf: str) -> tuple[str, bool]:
    """Return (new_confidence, promoted_candidate).

    Rules (per CLAUDE.md):
      1 → low
      3+ → medium
      5+ and no contradictions → still medium, flagged promoted_candidate (user confirms high)
      old_conf == 'high' stays high.
    """
    if old_conf == "high":
        return "high", False
    if new_count >= 3:
        new_conf = "medium"
    else:
        new_conf = "low"
    promoted = new_count >= 5 and not has_contradictions
    return new_conf, promoted


# ----------------------------- cascade-update -----------------------------


def _read_concept(slug: str) -> tuple[Path, dict[str, Any], str, str]:
    path = CONCEPTS_DIR / f"{slug}.md"
    if not path.is_file():
        _die(f"concept not found: {path.relative_to(REPO_ROOT)}")
    text = path.read_text(encoding="utf-8")
    fm_text, body = _split_frontmatter(text)
    fm = _parse_fm(fm_text)
    return path, fm, fm_text, body


def cmd_cascade_update(args: argparse.Namespace) -> int:
    source_slug = args.source
    bumped: list[dict[str, Any]] = []
    created: list[str] = []
    promoted: list[str] = []

    notes_map: dict[str, str] = {}
    for raw in args.note or []:
        if ":" not in raw:
            _die(f"--note expects concept:msg, got {raw!r}")
        k, _, v = raw.partition(":")
        notes_map[k.strip()] = v.strip()

    today = _today()
    source_link = f"- [[{source_slug}]]"

    for slug in args.bump or []:
        path, fm, fm_text, body = _read_concept(slug)
        old_count = int(fm.get("source_count") or 0)
        old_conf = str(fm.get("confidence") or "low")

        # Idempotent: if source_link already in ## Sources, skip.
        sources_text = _section_text(body, "Sources")
        if re.search(rf"^- \[\[{re.escape(source_slug)}\]\]\s*$", sources_text, re.MULTILINE):
            continue

        new_count = old_count + 1
        has_contradictions = _section_has_content(body, "Contradictions")
        new_conf, is_promoted = _new_confidence(new_count, has_contradictions, old_conf)

        new_fm_text = fm_text
        new_fm_text = _set_fm_field(new_fm_text, "source_count", new_count)
        new_fm_text = _set_fm_field(new_fm_text, "confidence", new_conf)
        new_fm_text = _set_fm_field(new_fm_text, "updated", today)
        new_fm_text = _set_fm_field(new_fm_text, "last_reviewed", today)

        new_body, _ = _append_to_section(body, "Sources", source_link, dedupe=True)
        note = notes_map.get(slug, "strengthened")
        ev_line = f"- {today} ({new_count} sources): {note}"
        new_body, _ = _append_to_section(new_body, "Evolution Log", ev_line, dedupe=False)

        _write_with_fm(path, new_fm_text, new_body)
        bumped.append({
            "slug": slug,
            "old_confidence": old_conf,
            "new_confidence": new_conf,
            "old_source_count": old_count,
            "new_source_count": new_count,
            "promoted_candidate": is_promoted,
        })
        if is_promoted:
            promoted.append(slug)

    for slug in args.create or []:
        path, fm, fm_text, body = _read_concept(slug)
        sc = fm.get("source_count")
        conf = fm.get("confidence")
        if int(sc or 0) != 1 or str(conf) != "low":
            print(
                f"wiki_ops: warning: --create {slug} expected source_count:1 confidence:low, "
                f"got source_count:{sc} confidence:{conf}",
                file=sys.stderr,
            )
        note = notes_map.get(slug, "created")
        ev_line = f"- {today} (1 source): {note}"
        sources_text = _section_text(body, "Sources")
        if not re.search(rf"^- \[\[{re.escape(source_slug)}\]\]\s*$", sources_text, re.MULTILINE):
            body, _ = _append_to_section(body, "Sources", f"- [[{source_slug}]]", dedupe=True)
        # Only add Evolution Log entry if log is empty (first entry).
        if not _section_has_content(body, "Evolution Log"):
            body, _ = _append_to_section(body, "Evolution Log", ev_line, dedupe=False)
        _write_with_fm(path, fm_text, body)
        created.append(slug)

    result = {"bumped": bumped, "created": created, "promoted_candidates": promoted}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# ----------------------------- index-update -----------------------------


def _replace_section(text: str, title: str, new_content: str) -> str:
    """Replace a section's body with `new_content` (caller controls leading/trailing newlines)."""
    s, e = _find_section(text, title)
    if s < 0:
        _die(f"index section missing: ## {title}")
    return text[:s] + new_content + text[e:]


def cmd_index_update(args: argparse.Namespace) -> int:
    if not INDEX_PATH.is_file():
        _die(f"{INDEX_PATH.relative_to(REPO_ROOT)} is missing")
    text = INDEX_PATH.read_text(encoding="utf-8")

    if args.add_paper:
        date = args.date or _today()
        slug = args.add_paper
        title = args.title or ""
        line = f"- `{date}` wiki/sources/{slug}.md — {title}".rstrip()

        # Remove from Unprocessed (any line mentioning the slug).
        up_s, up_e = _find_section(text, "Unprocessed")
        if up_s >= 0:
            up = text[up_s:up_e]
            cleaned = re.sub(
                rf"^- .*raw/.*{re.escape(slug)}.*$\n?", "", up, flags=re.MULTILINE
            )
            text = text[:up_s] + cleaned + text[up_e:]

        # Insert under ### Papers (under ## Sources), date-ordered: newest first.
        papers_m = re.search(r"^### Papers\s*$", text, re.MULTILINE)
        if not papers_m:
            _die("index missing '### Papers' subsection")
        insert_pos = papers_m.end()
        # Skip any leading blank or comment lines, find first entry or next ### heading.
        tail = text[insert_pos:]
        # Normalize: we simply inject the new line right after '### Papers' + one blank line.
        existing_line_re = re.compile(
            rf"^- .*wiki/sources/{re.escape(slug)}\.md.*$", re.MULTILINE
        )
        if existing_line_re.search(text):
            text = existing_line_re.sub(line, text, count=1)
        else:
            # Find existing bullets under Papers
            bullet_start = insert_pos
            # Skip whitespace + comment lines
            m = re.match(r"\s*(?:<!--.*?-->\s*)?", text[insert_pos:], re.DOTALL)
            if m:
                bullet_start = insert_pos + m.end()
            text = text[:bullet_start] + line + "\n" + text[bullet_start:]
            # Ensure a blank line after the section header if we ate it
            # (Harmless if already has one.)

    elif args.add_synthesis:
        slug = args.add_synthesis
        summary = args.summary or ""
        date = args.date or _today()
        line = f"- `{date}` — [[{slug}]] — {summary}".rstrip()
        new_body, appended = _append_to_section(text, "Recent Synthesis", line, dedupe=False)
        text = new_body

    elif args.resolve_question:
        q_text = args.resolve_question
        out_slug = args.output or ""
        if not QUESTIONS_PATH.is_file():
            _die(f"{QUESTIONS_PATH.relative_to(REPO_ROOT)} is missing")
        qtext = QUESTIONS_PATH.read_text(encoding="utf-8")
        open_s, open_e = _find_section(qtext, "Open Questions")
        if open_s < 0:
            _die("QUESTIONS.md missing '## Open Questions'")
        open_body = qtext[open_s:open_e]

        # Match a line containing the question text (case-insensitive, trimmed)
        needle = q_text.strip().lower()
        match_line = None
        for line in open_body.splitlines():
            if needle in line.lower():
                match_line = line
                break
        if match_line is None:
            _die(f"open question not found matching: {q_text!r}")
        # Remove the matched line from Open Questions.
        new_open = open_body.replace(match_line + "\n", "", 1)
        qtext = qtext[:open_s] + new_open + qtext[open_e:]

        # Construct resolved entry
        today = _today()
        resolved_line = match_line.strip()
        # Convert `- [ ] <q> (opened YYYY-MM-DD)` → `- [x] <q> (opened YYYY-MM-DD, resolved YYYY-MM-DD → wiki/outputs/<slug>.md)`
        resolved_line = re.sub(r"^- \[ \]", "- [x]", resolved_line)
        if re.search(r"\(opened [\d-]+", resolved_line):
            resolved_line = re.sub(
                r"(\(opened [\d-]+)\)",
                rf"\1, resolved {today} → wiki/outputs/{out_slug}.md)",
                resolved_line,
                count=1,
            )
        else:
            resolved_line = f"{resolved_line} (resolved {today} → wiki/outputs/{out_slug}.md)"
        qtext, _ = _append_to_section(qtext, "Resolved Questions", resolved_line, dedupe=False)
        QUESTIONS_PATH.write_text(qtext, encoding="utf-8")
        return 0
    else:
        _die("index-update requires one of --add-paper / --add-synthesis / --resolve-question")

    INDEX_PATH.write_text(text, encoding="utf-8")
    return 0


# ----------------------------- log-append -----------------------------


def cmd_log_append(args: argparse.Namespace) -> int:
    if not LOG_PATH.is_file():
        _die(f"{LOG_PATH.relative_to(REPO_ROOT)} is missing")
    line = f"{_today()} {_now_hhmm()} | {args.op} | {args.subject}\n"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)
    return 0


# ----------------------------- rejection-append -----------------------------


REJECTIONS_HEADER = """---
type: rejection-log
graph-excluded: true
append-only: true
date: {today}
---

# Rejection Log

Append-only record of user-rejected drafts (ingest, query, merge, …). Newest first.

"""


def cmd_rejection_append(args: argparse.Namespace) -> int:
    today = _today()
    now_stamp = f"{today} {_now_hhmm()}"
    if not REJECTIONS_PATH.is_file():
        REJECTIONS_PATH.write_text(REJECTIONS_HEADER.format(today=today), encoding="utf-8")

    entry_lines = [
        f"## {now_stamp} — {args.op} — {args.subject}",
        f"- **Drafted**: {args.drafted or '(see context)'}",
        f"- **Rejected reason**: {args.reason}",
    ]
    if args.context:
        entry_lines.append(f"- **Context**: {args.context}")
    entry_lines.append("")
    entry = "\n".join(entry_lines) + "\n"

    text = REJECTIONS_PATH.read_text(encoding="utf-8")
    # Insert after the `# Rejection Log` heading + its blurb, before first `## ` (if any).
    m = re.search(r"^# Rejection Log\s*$", text, re.MULTILINE)
    if not m:
        # Fallback: append at end
        REJECTIONS_PATH.write_text(text.rstrip() + "\n\n" + entry, encoding="utf-8")
        return 0
    head_end = m.end()
    # Find first line starting with "## " after head_end, or end of file
    tail = text[head_end:]
    m2 = re.search(r"^## ", tail, re.MULTILINE)
    insert_pos = head_end + (m2.start() if m2 else len(tail))
    # Ensure a blank line before entry if needed
    left = text[:insert_pos].rstrip() + "\n\n"
    REJECTIONS_PATH.write_text(left + entry + text[insert_pos:].lstrip(), encoding="utf-8")
    return 0


# ----------------------------- merge-execute -----------------------------


CONF_ORDER = {"low": 1, "medium": 2, "high": 3}


def _max_confidence(a: str, b: str) -> str:
    if CONF_ORDER.get(a, 0) >= CONF_ORDER.get(b, 0):
        return a or "low"
    return b or "low"


def _union_list(a: Iterable, b: Iterable) -> list:
    out, seen = [], set()
    for item in list(a or []) + list(b or []):
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _union_bullets(a_section: str, b_section: str) -> str:
    lines_a = [ln for ln in a_section.splitlines() if ln.strip()]
    lines_b = [ln for ln in b_section.splitlines() if ln.strip()]
    seen, out = set(), []
    for ln in lines_a + lines_b:
        if HTML_COMMENT_RE.fullmatch(ln.strip() or "xxx"):
            continue
        key = ln.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(ln)
    return "\n".join(out) + ("\n" if out else "")


def _merge_evolution_logs(a: str, b: str, append_merge_note: str) -> str:
    entries = []
    for section in (a, b):
        for ln in section.splitlines():
            s = ln.strip()
            if not s or HTML_COMMENT_RE.fullmatch(s):
                continue
            m = re.match(r"^- (\d{4}-\d{2}-\d{2})\b", s)
            key = m.group(1) if m else "0000-00-00"
            entries.append((key, ln))
    entries.sort(key=lambda t: t[0])
    dedup = []
    seen = set()
    for _, ln in entries:
        if ln not in seen:
            seen.add(ln)
            dedup.append(ln)
    dedup.append(append_merge_note)
    return "\n".join(dedup) + "\n"


def cmd_merge_execute(args: argparse.Namespace) -> int:
    keep = args.keep
    drop = args.drop
    if keep == drop:
        _die("--keep and --drop must differ")

    keep_path, keep_fm, keep_fm_text, keep_body = _read_concept(keep)
    drop_path, drop_fm, drop_fm_text, drop_body = _read_concept(drop)

    today = _today()

    # Merged frontmatter
    new_aliases = _union_list(keep_fm.get("aliases") or [], drop_fm.get("aliases") or [])
    new_source_count = int(keep_fm.get("source_count") or 0) + int(drop_fm.get("source_count") or 0)
    new_confidence = _max_confidence(
        str(keep_fm.get("confidence") or "low"),
        str(drop_fm.get("confidence") or "low"),
    )
    dates = [d for d in (keep_fm.get("date"), drop_fm.get("date")) if d]
    new_date = min(str(d) for d in dates) if dates else today

    fm_patched = keep_fm_text
    fm_patched = _set_fm_field(fm_patched, "aliases", new_aliases)
    fm_patched = _set_fm_field(fm_patched, "source_count", new_source_count)
    fm_patched = _set_fm_field(fm_patched, "confidence", new_confidence)
    fm_patched = _set_fm_field(fm_patched, "date", new_date)
    fm_patched = _set_fm_field(fm_patched, "updated", today)
    fm_patched = _set_fm_field(fm_patched, "last_reviewed", today)

    # Merged body sections
    def _merged_section(title: str, merger=_union_bullets) -> str:
        return merger(_section_text(keep_body, title), _section_text(drop_body, title))

    new_body = keep_body  # start with keep
    for title in ("Key Points", "My Position", "Contradictions", "Sources"):
        merged = _merged_section(title)
        s, e = _find_section(new_body, title)
        if s < 0:
            # Append section if missing
            sep = "" if new_body.endswith("\n") else "\n"
            new_body = new_body + sep + f"\n## {title}\n\n{merged}\n"
        else:
            new_body = new_body[:s] + "\n" + merged + "\n" + new_body[e:]

    # Evolution Log: chronological merge + append merge note
    merge_note = f"- {today} merged [[{drop}]] into this page"
    ev_merged = _merge_evolution_logs(
        _section_text(keep_body, "Evolution Log"),
        _section_text(drop_body, "Evolution Log"),
        merge_note,
    )
    s, e = _find_section(new_body, "Evolution Log")
    if s < 0:
        new_body = new_body.rstrip() + f"\n\n## Evolution Log\n\n{ev_merged}"
    else:
        new_body = new_body[:s] + "\n" + ev_merged + new_body[e:]

    # Scan wiki/ for [[drop]] wikilinks (preserve |display alias where present).
    rewrites: list[tuple[Path, int]] = []
    link_re = re.compile(rf"\[\[{re.escape(drop)}((?:\|[^\]]+)?)\]\]")
    for p in WIKI_DIR.rglob("*.md"):
        if p == drop_path or p == keep_path:
            continue
        text = p.read_text(encoding="utf-8")
        if link_re.search(text):
            count = len(link_re.findall(text))
            rewrites.append((p, count))

    if args.dry_run or not args.confirmed:
        plan = {
            "keep": keep,
            "drop": drop,
            "new_aliases": new_aliases,
            "new_source_count": new_source_count,
            "new_confidence": new_confidence,
            "wikilink_rewrites": [
                {"file": str(p.relative_to(REPO_ROOT)), "count": c} for p, c in rewrites
            ],
            "drop_page_becomes_redirect": True,
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("\n(dry-run: pass --confirmed to execute)", file=sys.stderr)
        return 0

    # Execute
    _write_with_fm(keep_path, fm_patched, new_body)

    # Drop page → redirect stub
    drop_alias_list = _union_list(drop_fm.get("aliases") or [], [])
    stub_fm = (
        f"type: concept\ntitle: {drop_fm.get('title', drop)}\nslug: {drop}\n"
        f"redirect: {keep}\naliases: {yaml.safe_dump(drop_alias_list, default_flow_style=True).strip()}\n"
        f"date: {today}\n"
    )
    stub_body = f"\nThis page has been merged. See [[{keep}]].\n"
    _write_with_fm(drop_path, stub_fm, stub_body)

    for p, _ in rewrites:
        text = p.read_text(encoding="utf-8")
        text = link_re.sub(rf"[[{keep}\1]]", text)
        p.write_text(text, encoding="utf-8")

    print(json.dumps({"status": "merged", "keep": keep, "drop": drop, "rewritten_files": len(rewrites)}, indent=2))
    return 0


# ----------------------------- CLI -----------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="MyWiki unified bookkeeping CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cascade-update", help="strengthen/create concept pages after an INGEST")
    p.add_argument("--source", required=True, help="source slug (without .md)")
    p.add_argument("--bump", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=[],
                   help="comma-separated concept slugs to strengthen")
    p.add_argument("--create", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=[],
                   help="comma-separated concept slugs that were just created by LLM")
    p.add_argument("--note", action="append", default=[],
                   help="custom Evolution Log note for a concept, format: concept:msg (repeatable)")
    p.set_defaults(func=cmd_cascade_update)

    p = sub.add_parser("index-update", help="mutate wiki/index.md sections")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--add-paper", metavar="SLUG")
    g.add_argument("--add-synthesis", metavar="OUTPUT-SLUG")
    g.add_argument("--resolve-question", metavar="QUESTION-SUBSTRING")
    p.add_argument("--title", default="")
    p.add_argument("--summary", default="")
    p.add_argument("--date", default="")
    p.add_argument("--output", default="", help="output slug for --resolve-question")
    p.set_defaults(func=cmd_index_update)

    p = sub.add_parser("log-append", help="append a timestamped entry to wiki/log.md")
    p.add_argument("op", help="operation name (ingest, query, lint, ...)")
    p.add_argument("subject", help="subject / target description")
    p.set_defaults(func=cmd_log_append)

    p = sub.add_parser("rejection-append", help="append a user-rejection entry to wiki/rejections.md")
    p.add_argument("--op", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--context", default="")
    p.add_argument("--drafted", default="")
    p.set_defaults(func=cmd_rejection_append)

    p = sub.add_parser("merge-execute", help="merge concept <drop> into <keep> (dry-run default)")
    p.add_argument("--keep", required=True)
    p.add_argument("--drop", required=True)
    p.add_argument("--dry-run", action="store_true", help="show plan without writing")
    p.add_argument("--confirmed", action="store_true", help="actually execute; without it, dry-run")
    p.set_defaults(func=cmd_merge_execute)

    args = parser.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
