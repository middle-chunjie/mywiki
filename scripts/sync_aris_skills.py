#!/usr/bin/env python3
"""Selective ARIS skill synchronization for MyWiki.

Syncs specific skills from the upstream ARIS repository while preserving
local customizations (DIY forks) and purely local skills (wiki-specific).

Each skill is tracked in a manifest (.claude/aris-sync.json) with one of
three modes:
  - track   : follow upstream; auto-update unless local edits detected
  - fork    : initially from ARIS, now independently maintained (never overwrite)
  - local   : purely local skill, invisible to sync

Subcommands:
  status       show sync state for all managed skills
  check        dry-run: compare tracked skills against upstream, report diffs
  apply        apply safe updates (new + unmodified tracked skills)
  add          start tracking a new ARIS skill
  fork         convert a tracked skill to fork mode (stop syncing)
  unfork       convert a forked skill back to track mode
  diff         show diff between local and upstream for a specific skill

Requires: git (for cloning/pulling upstream repo)

Usage:
  python scripts/sync_aris_skills.py status
  python scripts/sync_aris_skills.py check
  python scripts/sync_aris_skills.py apply
  python scripts/sync_aris_skills.py add idea-discovery research-refine
  python scripts/sync_aris_skills.py fork idea-discovery
  python scripts/sync_aris_skills.py diff idea-discovery
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_URL = "https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git"
UPSTREAM_SKILLS_SUBDIR = "skills"

WIKI_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = WIKI_ROOT / ".claude" / "aris-sync.json"
LOCAL_SKILLS_DIR = Path.home() / ".claude" / "skills"
ARCHIVE_DIR = Path.home() / ".claude" / "skills-archive"

# Skills that should never be synced or installed (user preference)
EXCLUDED_SKILLS = {
    # Patent skills — user confirmed not needed (2026-04-21)
    "claims-drafting", "patent-pipeline", "patent-review", "patent-novelty-check",
    "prior-art-search", "invention-structuring", "specification-writing",
    "embodiment-description", "figure-description", "jurisdiction-format",
}

# Patterns indicating personal customization (from ARIS smart_update)
PERSONAL_PATTERNS = [
    "ssh ", "api_key", "sk-", "192.168.", "10.0.", "@sjtu",
    "/home/", "/Users/", "/root/", "CUDA_VISIBLE", "conda activate",
    "api_token", "secret", "password", "credential",
]


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"upstream_repo": REPO_URL, "last_sync": None, "skills": {}}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest["last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _hash_dir(path: Path) -> str:
    """Content hash of all files in a directory (sorted, deterministic)."""
    h = hashlib.sha256()
    if not path.exists():
        return ""
    for f in sorted(path.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _has_personal_patterns(path: Path) -> list[str]:
    """Scan skill directory for personal customization patterns."""
    found = []
    if not path.exists():
        return found
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        try:
            content = f.read_text(errors="ignore").lower()
        except Exception:
            continue
        for pat in PERSONAL_PATTERNS:
            if pat.lower() in content:
                found.append(f"{f.name}: contains '{pat}'")
    return found


def _clone_or_pull_upstream(cache_dir: Path) -> Path:
    """Clone or pull the ARIS repo into a cache directory."""
    repo_dir = cache_dir / "aris-upstream"
    if repo_dir.exists() and (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--quiet"],
            capture_output=True,
        )
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", REPO_URL, str(repo_dir)],
            capture_output=True,
            check=True,
        )
    return repo_dir / UPSTREAM_SKILLS_SUBDIR


def _get_upstream_skills_dir() -> Path:
    """Get the upstream skills directory, using a persistent cache."""
    cache_dir = Path.home() / ".cache" / "mywiki-aris-sync"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _clone_or_pull_upstream(cache_dir)


def _list_upstream_skills(upstream_dir: Path) -> list[str]:
    """List available skill names in upstream."""
    if not upstream_dir.exists():
        return []
    return sorted(
        d.name for d in upstream_dir.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    )


# ── Subcommands ──────────────────────────────────────────────────────

def cmd_status(args):
    """Show sync state for all managed skills."""
    manifest = _load_manifest()
    skills = manifest.get("skills", {})

    if not skills:
        print("No skills tracked yet. Use 'add <skill-name>' to start tracking.")
        return

    print(f"{'Skill':<35} {'Mode':<8} {'Local Hash':<18} {'Synced Hash':<18} {'Status'}")
    print("-" * 100)

    for name, info in sorted(skills.items()):
        mode = info.get("mode", "track")
        synced_hash = info.get("synced_hash", "")[:16]
        local_path = LOCAL_SKILLS_DIR / name
        local_hash = _hash_dir(local_path)[:16] if local_path.exists() else "(missing)"

        if mode == "fork":
            status = "forked (no sync)"
        elif mode == "local":
            status = "local-only"
        elif local_hash == "(missing)":
            status = "NOT INSTALLED"
        elif local_hash == synced_hash:
            status = "up-to-date"
        else:
            status = "LOCAL EDITS"

        print(f"{name:<35} {mode:<8} {local_hash:<18} {synced_hash:<18} {status}")

    print(f"\nLast sync: {manifest.get('last_sync', 'never')}")


def cmd_check(args):
    """Dry-run: compare tracked skills against upstream."""
    manifest = _load_manifest()
    skills = manifest.get("skills", {})

    print("Fetching upstream...")
    upstream_dir = _get_upstream_skills_dir()
    upstream_available = _list_upstream_skills(upstream_dir)

    categories = {
        "identical": [],
        "new_upstream": [],
        "safe_update": [],
        "needs_merge": [],
        "local_only": [],
        "not_tracked": [],
    }

    for name, info in sorted(skills.items()):
        mode = info.get("mode", "track")
        if mode in ("fork", "local"):
            categories["local_only"].append((name, mode))
            continue

        local_path = LOCAL_SKILLS_DIR / name
        upstream_path = upstream_dir / name

        if not upstream_path.exists():
            categories["local_only"].append((name, "upstream-removed"))
            continue

        local_hash = _hash_dir(local_path)
        upstream_hash = _hash_dir(upstream_path)
        synced_hash = info.get("synced_hash", "")

        if local_hash == upstream_hash:
            categories["identical"].append(name)
        elif local_hash == synced_hash:
            # Local hasn't changed since last sync → safe to update
            categories["safe_update"].append(name)
        else:
            # Both local and upstream changed
            personal = _has_personal_patterns(local_path)
            categories["needs_merge"].append((name, personal))

    # Report new skills available upstream but not tracked
    tracked_names = set(skills.keys())
    for name in upstream_available:
        if name not in tracked_names and name not in EXCLUDED_SKILLS:
            categories["not_tracked"].append(name)

    # Print report
    print(f"\n{'='*60}")
    print(f"ARIS Skill Sync Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*60}")

    print(f"\n✓ Identical (no action needed): {len(categories['identical'])}")
    for name in categories["identical"]:
        print(f"  {name}")

    if categories["safe_update"]:
        print(f"\n⬆ Safe to update: {len(categories['safe_update'])}")
        for name in categories["safe_update"]:
            print(f"  {name}")

    if categories["needs_merge"]:
        print(f"\n⚠ Needs manual merge: {len(categories['needs_merge'])}")
        for name, personal in categories["needs_merge"]:
            print(f"  {name}")
            if personal:
                for p in personal[:3]:
                    print(f"    → {p}")

    if categories["local_only"]:
        print(f"\n🔒 Local/forked (preserved): {len(categories['local_only'])}")
        for name, mode in categories["local_only"]:
            print(f"  {name} ({mode})")

    if categories["not_tracked"]:
        print(f"\n📦 Available upstream (not tracked): {len(categories['not_tracked'])}")
        for name in categories["not_tracked"][:20]:
            print(f"  {name}")
        if len(categories["not_tracked"]) > 20:
            print(f"  ... and {len(categories['not_tracked']) - 20} more")

    return categories


def cmd_apply(args):
    """Apply safe updates from upstream."""
    categories = cmd_check(args)
    manifest = _load_manifest()

    safe = categories.get("safe_update", [])
    if not safe:
        print("\nNothing to update.")
        return

    print(f"\nApplying {len(safe)} safe update(s)...")
    upstream_dir = _get_upstream_skills_dir()

    for name in safe:
        src = upstream_dir / name
        dst = LOCAL_SKILLS_DIR / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        new_hash = _hash_dir(dst)
        manifest["skills"][name]["synced_hash"] = new_hash
        manifest["skills"][name]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        print(f"  ✓ {name}")

    _save_manifest(manifest)
    print(f"\nDone. {len(safe)} skill(s) updated.")


def cmd_add(args):
    """Start tracking one or more ARIS skills."""
    manifest = _load_manifest()
    names = args.names

    print("Fetching upstream...")
    upstream_dir = _get_upstream_skills_dir()
    available = _list_upstream_skills(upstream_dir)

    added = []
    for name in names:
        if name in EXCLUDED_SKILLS:
            print(f"  ✗ {name} is in the exclusion list (EXCLUDED_SKILLS). Use --force to override.")
            continue

        if name in manifest.get("skills", {}):
            print(f"  ⏭ {name} already tracked (mode: {manifest['skills'][name]['mode']})")
            continue

        if name not in available:
            print(f"  ✗ {name} not found in upstream. Available: {', '.join(available[:10])}...")
            continue

        # Install if not present locally
        local_path = LOCAL_SKILLS_DIR / name
        upstream_path = upstream_dir / name
        if not local_path.exists():
            shutil.copytree(upstream_path, local_path)
            print(f"  ✓ {name} installed from upstream")
        else:
            print(f"  ✓ {name} already installed locally, now tracking")

        current_hash = _hash_dir(local_path)
        manifest.setdefault("skills", {})[name] = {
            "mode": "track",
            "synced_hash": current_hash,
            "added": datetime.now().strftime("%Y-%m-%d"),
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
        }
        added.append(name)

    _save_manifest(manifest)
    if added:
        print(f"\nAdded {len(added)} skill(s) to tracking: {', '.join(added)}")


def cmd_fork(args):
    """Convert a tracked skill to fork mode (stop syncing)."""
    manifest = _load_manifest()
    name = args.name

    if name not in manifest.get("skills", {}):
        print(f"Skill '{name}' is not tracked. Use 'add' first.")
        return

    manifest["skills"][name]["mode"] = "fork"
    manifest["skills"][name]["forked_date"] = datetime.now().strftime("%Y-%m-%d")
    _save_manifest(manifest)
    print(f"✓ '{name}' is now forked. It will be preserved during sync and never overwritten.")


def cmd_unfork(args):
    """Convert a forked skill back to track mode."""
    manifest = _load_manifest()
    name = args.name

    if name not in manifest.get("skills", {}):
        print(f"Skill '{name}' is not tracked.")
        return

    if manifest["skills"][name].get("mode") != "fork":
        print(f"Skill '{name}' is not in fork mode (current: {manifest['skills'][name]['mode']})")
        return

    local_hash = _hash_dir(LOCAL_SKILLS_DIR / name)
    manifest["skills"][name]["mode"] = "track"
    manifest["skills"][name]["synced_hash"] = local_hash
    manifest["skills"][name].pop("forked_date", None)
    _save_manifest(manifest)
    print(f"✓ '{name}' is now tracked again. Current local version set as sync baseline.")


def cmd_diff(args):
    """Show diff between local and upstream for a specific skill."""
    name = args.name
    local_path = LOCAL_SKILLS_DIR / name
    if not local_path.exists():
        print(f"Local skill '{name}' not found at {local_path}")
        return

    print("Fetching upstream...")
    upstream_dir = _get_upstream_skills_dir()
    upstream_path = upstream_dir / name
    if not upstream_path.exists():
        print(f"Upstream skill '{name}' not found.")
        return

    result = subprocess.run(
        ["diff", "-ru", str(upstream_path), str(local_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("No differences.")
    else:
        print(result.stdout)


def cmd_list_upstream(args):
    """List all available skills in upstream repo."""
    print("Fetching upstream...")
    upstream_dir = _get_upstream_skills_dir()
    available = _list_upstream_skills(upstream_dir)
    manifest = _load_manifest()
    tracked = set(manifest.get("skills", {}).keys())

    excluded_count = sum(1 for n in available if n in EXCLUDED_SKILLS)
    active = [n for n in available if n not in EXCLUDED_SKILLS]
    print(f"\nAvailable ARIS skills ({len(active)} eligible, {excluded_count} excluded):\n")
    for name in active:
        marker = ""
        if name in tracked:
            mode = manifest["skills"][name].get("mode", "track")
            marker = f" [{mode}]"
        print(f"  {name}{marker}")
    if excluded_count:
        print(f"\n  ({excluded_count} patent skills excluded — edit EXCLUDED_SKILLS to change)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Selective ARIS skill synchronization for MyWiki"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show sync state for all managed skills")
    sub.add_parser("check", help="Dry-run: compare tracked skills against upstream")
    sub.add_parser("apply", help="Apply safe updates from upstream")

    p_add = sub.add_parser("add", help="Start tracking ARIS skill(s)")
    p_add.add_argument("names", nargs="+", help="Skill name(s) to track")

    p_fork = sub.add_parser("fork", help="Convert tracked skill to fork mode")
    p_fork.add_argument("name", help="Skill name to fork")

    p_unfork = sub.add_parser("unfork", help="Convert forked skill back to track mode")
    p_unfork.add_argument("name", help="Skill name to unfork")

    p_diff = sub.add_parser("diff", help="Show diff for a specific skill")
    p_diff.add_argument("name", help="Skill name to diff")

    sub.add_parser("list-upstream", help="List all available upstream skills")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "status": cmd_status,
        "check": cmd_check,
        "apply": cmd_apply,
        "add": cmd_add,
        "fork": cmd_fork,
        "unfork": cmd_unfork,
        "diff": cmd_diff,
        "list-upstream": cmd_list_upstream,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
