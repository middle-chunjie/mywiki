#!/usr/bin/env python3
"""Project lifecycle CLI for MyWiki research projects.

Manages projects/<slug>/ directories: creation, stage advancement, status
queries, and archival. Modeled on wiki_ops.py — same frontmatter helpers,
same JSON output convention.

Subcommands:
  create          scaffold a new project directory tree
  status          show JSON status for one or all projects
  advance-stage   move a project to the next research stage
  list            compact one-line-per-project listing
  archive         set stage=archived
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiki_ops import (
    _split_frontmatter,
    _parse_fm,
    _set_fm_field,
    _write_with_fm,
    _today,
    _append_to_section,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = REPO_ROOT / "projects"
TEMPLATE_DIR = REPO_ROOT / "wiki" / "templates"

STAGES = [
    "created",
    "lit-review",
    "ideation",
    "refinement",
    "planning",
    "implementation",
    "experimentation",
    "review",
    "writing",
    "submitted",
    "archived",
]

STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}

VALID_TRANSITIONS: dict[str, set[str]] = {}
for i, s in enumerate(STAGES[:-1]):
    VALID_TRANSITIONS.setdefault(s, set()).add(STAGES[i + 1])
# Pivots
VALID_TRANSITIONS.setdefault("review", set()).update({"planning", "ideation"})
VALID_TRANSITIONS.setdefault("experimentation", set()).update({"implementation", "planning"})
VALID_TRANSITIONS.setdefault("refinement", set()).add("ideation")
# Any non-archived stage can archive
for s in STAGES:
    if s != "archived":
        VALID_TRANSITIONS.setdefault(s, set()).add("archived")


def _die(msg: str, code: int = 1) -> None:
    print(f"project_ops: {msg}", file=sys.stderr)
    sys.exit(code)


def _project_dir(slug: str) -> Path:
    return PROJECTS_DIR / slug


def _project_md(slug: str) -> Path:
    return _project_dir(slug) / "PROJECT.md"


def _read_project(slug: str) -> tuple[Path, dict, str, str]:
    path = _project_md(slug)
    if not path.is_file():
        _die(f"project not found: {slug}")
    text = path.read_text(encoding="utf-8")
    fm_text, body = _split_frontmatter(text)
    fm = _parse_fm(fm_text)
    return path, fm, fm_text, body


# ----------------------------- create -----------------------------


SUBDIRS = [
    "lit-review",
    "idea-stage",
    "refine-logs",
    "experiments",
    "review-stage",
    "paper",
    "wiki-contributions",
]

PENDING_FILES = [
    "wiki-contributions/pending-sources.md",
    "wiki-contributions/pending-concepts.md",
    "wiki-contributions/pending-synthesis.md",
]


def cmd_create(args: argparse.Namespace) -> int:
    slug = args.slug
    direction = args.direction
    domains = [d.strip() for d in (args.domains or "").split(",") if d.strip()]

    pdir = _project_dir(slug)
    if pdir.exists():
        _die(f"project already exists: {slug}")

    today = _today()

    # Create directory tree
    for sub in SUBDIRS:
        (pdir / sub).mkdir(parents=True, exist_ok=True)

    # Render PROJECT.md from template
    tpl = (TEMPLATE_DIR / "project-template.md").read_text(encoding="utf-8")
    fm_text, body = _split_frontmatter(tpl)
    fm_text = _set_fm_field(fm_text, "title", direction[:80])
    fm_text = _set_fm_field(fm_text, "slug", slug)
    fm_text = _set_fm_field(fm_text, "direction", direction)
    fm_text = _set_fm_field(fm_text, "created", today)
    fm_text = _set_fm_field(fm_text, "updated", today)
    if domains:
        fm_text = _set_fm_field(fm_text, "wiki_read_domains", domains)
    _write_with_fm(pdir / "PROJECT.md", fm_text, body)

    # Init pending files
    for pf in PENDING_FILES:
        pfpath = pdir / pf
        kind = pf.split("-", 1)[1].replace(".md", "")  # sources, concepts, synthesis
        pf_fm = f"type: pending-writeback\nproject: {slug}\nkind: {kind}\ndate: {today}\n"
        pf_body = f"\n# Pending {kind.title()}\n\nItems discovered during research, staged for wiki writeback.\n"
        _write_with_fm(pfpath, pf_fm, pf_body)

    result = {
        "status": "created",
        "slug": slug,
        "direction": direction,
        "path": str(pdir.relative_to(REPO_ROOT)),
        "domains": domains,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# ----------------------------- status -----------------------------


def _project_status_dict(slug: str) -> dict:
    path, fm, _, _ = _read_project(slug)
    from datetime import date

    updated_str = str(fm.get("updated") or fm.get("created") or "")
    days_since = -1
    if updated_str:
        try:
            updated_date = date.fromisoformat(updated_str)
            days_since = (date.today() - updated_date).days
        except ValueError:
            pass

    return {
        "slug": slug,
        "title": fm.get("title", ""),
        "stage": fm.get("stage", "unknown"),
        "direction": fm.get("direction", ""),
        "updated": updated_str,
        "days_since_update": days_since,
    }


def cmd_status(args: argparse.Namespace) -> int:
    if args.slug:
        if not _project_md(args.slug).is_file():
            _die(f"project not found: {args.slug}")
        result = [_project_status_dict(args.slug)]
    else:
        if not PROJECTS_DIR.is_dir():
            result = []
        else:
            result = []
            for p in sorted(PROJECTS_DIR.iterdir()):
                if p.is_dir() and (p / "PROJECT.md").is_file():
                    result.append(_project_status_dict(p.name))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# ----------------------------- advance-stage -----------------------------


def cmd_advance_stage(args: argparse.Namespace) -> int:
    slug = args.slug
    target = args.to

    if target not in STAGE_INDEX:
        _die(f"unknown stage: {target}. Valid: {', '.join(STAGES)}")

    path, fm, fm_text, body = _read_project(slug)
    current = fm.get("stage", "created")

    if current not in VALID_TRANSITIONS:
        _die(f"no transitions defined from stage: {current}")
    if target not in VALID_TRANSITIONS[current]:
        allowed = ", ".join(sorted(VALID_TRANSITIONS[current]))
        _die(f"invalid transition: {current} → {target}. Allowed from {current}: {allowed}")

    today = _today()
    fm_text = _set_fm_field(fm_text, "stage", target)
    fm_text = _set_fm_field(fm_text, "updated", today)

    history_line = f"- {today}: {current} → {target}"
    body, _ = _append_to_section(body, "Stage History", history_line)
    _write_with_fm(path, fm_text, body)

    result = {
        "status": "advanced",
        "slug": slug,
        "from": current,
        "to": target,
        "date": today,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# ----------------------------- list -----------------------------


def cmd_list(args: argparse.Namespace) -> int:
    if not PROJECTS_DIR.is_dir():
        print("(no projects)")
        return 0
    found = False
    for p in sorted(PROJECTS_DIR.iterdir()):
        if p.is_dir() and (p / "PROJECT.md").is_file():
            found = True
            info = _project_status_dict(p.name)
            stale = " [STALE]" if info["days_since_update"] > 7 else ""
            print(f"  {info['slug']:30s} {info['stage']:20s} updated {info['updated']}{stale}")
    if not found:
        print("(no projects)")
    return 0


# ----------------------------- archive -----------------------------


def cmd_archive(args: argparse.Namespace) -> int:
    slug = args.slug
    path, fm, fm_text, body = _read_project(slug)
    current = fm.get("stage", "created")

    if current == "archived":
        _die(f"project {slug} is already archived")

    today = _today()
    fm_text = _set_fm_field(fm_text, "stage", "archived")
    fm_text = _set_fm_field(fm_text, "updated", today)

    history_line = f"- {today}: {current} → archived"
    body, _ = _append_to_section(body, "Stage History", history_line)
    _write_with_fm(path, fm_text, body)

    result = {"status": "archived", "slug": slug, "from": current, "date": today}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# ----------------------------- CLI -----------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="MyWiki project lifecycle CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create", help="scaffold a new project directory")
    p.add_argument("--slug", required=True, help="project slug (kebab-case)")
    p.add_argument("--direction", required=True, help="research direction (1-2 sentences)")
    p.add_argument("--domains", default="", help="comma-separated wiki domains to read")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("status", help="show project status (JSON)")
    p.add_argument("--slug", default="", help="specific project (default: all)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("advance-stage", help="move project to next stage")
    p.add_argument("--slug", required=True)
    p.add_argument("--to", required=True, help="target stage")
    p.set_defaults(func=cmd_advance_stage)

    p = sub.add_parser("list", help="compact one-line-per-project listing")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("archive", help="archive a project")
    p.add_argument("--slug", required=True)
    p.set_defaults(func=cmd_archive)

    args = parser.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
