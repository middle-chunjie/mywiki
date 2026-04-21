#!/usr/bin/env python3
"""Scan the current Claude Code session transcript for digest candidates.

Reads the most recent .jsonl in
  ~/.claude/projects/-Users-ruili-Documents-NoteLibrary-MyWiki/
filters messages newer than the last-digest timestamp, and emits:
  1. Hard candidates: arxiv IDs, DOIs, URLs (deduplicated, with turn context)
  2. Filtered transcript body (user + assistant text only, tool calls stripped)

The skill reads this output and applies judgment to classify into
  sources / concepts / positions / questions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/ruili/Documents/NoteLibrary/MyWiki")
PROJECT_DIR = Path.home() / ".claude" / "projects" / "-Users-ruili-Documents-NoteLibrary-MyWiki"
STATE_FILE = REPO / ".claude" / "digest-state.json"
SESSIONS_DIR = REPO / ".claude" / "sessions"
ANCESTRY_MAX_DEPTH = 20

ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)>\]`\"]+")
SKIP_DOMAINS = {
    "localhost", "127.0.0.1", "example.com",
    "github.com/anthropics/claude-code",
}


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_digest_ts": "2000-01-01T00:00:00Z"}


def _ppid_of(pid: int) -> int | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True, text=True, check=False, timeout=2,
        ).stdout.strip()
        return int(out) if out and out != "0" else None
    except Exception:
        return None


def resolve_current_session() -> tuple[Path | None, str]:
    """Walk up the process tree from this script's PID; for each ancestor,
    look for .claude/sessions/session-<PID>.json. Returns (transcript_path, note).

    The registry is written by the SessionStart `session-start-register.sh` hook,
    whose $PPID at hook time is the Claude Code process itself. So any skill/bash
    running inside that session has that PID in its ancestry.
    """
    if not SESSIONS_DIR.exists():
        return None, "no session registry (SessionStart register hook may not have run)"

    pid = os.getppid()
    for _ in range(ANCESTRY_MAX_DEPTH):
        if pid is None or pid <= 1:
            break
        reg = SESSIONS_DIR / f"session-{pid}.json"
        if reg.exists():
            try:
                data = json.loads(reg.read_text())
                tp = Path(data.get("transcript_path", ""))
                if tp.exists():
                    return tp, f"matched ancestor pid={pid}"
            except Exception:
                pass
        pid = _ppid_of(pid)

    return None, "no matching ancestor in session registry"


def find_latest_jsonl() -> Path | None:
    """Fallback only — returns the most recently modified jsonl in the project
    directory. Unsafe under concurrent sessions; callers should prefer
    resolve_current_session() or an explicit --transcript."""
    if not PROJECT_DIR.exists():
        return None
    files = sorted(PROJECT_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def resolve_transcript(explicit: str | None = None) -> tuple[Path | None, str]:
    """Unified resolver: explicit override > session registry > latest-jsonl fallback."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p, "explicit --transcript"
        return None, f"explicit transcript not found: {explicit}"
    tp, note = resolve_current_session()
    if tp:
        return tp, note
    fallback = find_latest_jsonl()
    if fallback:
        return fallback, f"FALLBACK (latest-mtime, unsafe for concurrent sessions): {note}"
    return None, f"no transcript found: {note}"


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return ""


def iter_messages(jsonl: Path, since_ts: str):
    with jsonl.open() as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = o.get("type")
            if t not in ("user", "assistant"):
                continue
            ts = o.get("timestamp", "")
            if ts <= since_ts:
                continue
            msg = o.get("message") or {}
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or t
            text = extract_text(msg.get("content"))
            if not text.strip():
                continue
            if text.lstrip().startswith("<local-command-"):
                continue
            if "<system-reminder>" in text and len(text) < 500:
                continue
            yield {
                "ts": ts,
                "uuid": o.get("uuid"),
                "role": role,
                "text": text,
            }


def scan(since_ts: str, transcript_path: str | None = None) -> dict:
    jsonl, note = resolve_transcript(transcript_path)
    if not jsonl:
        return {"error": note, "since_ts": since_ts}

    messages = list(iter_messages(jsonl, since_ts))
    arxiv_hits: dict[str, dict] = {}
    doi_hits: dict[str, dict] = {}
    url_hits: dict[str, dict] = {}

    for m in messages:
        text = m["text"]
        for match in ARXIV_RE.finditer(text):
            aid = match.group(1)
            arxiv_hits.setdefault(aid, {
                "id": aid, "role": m["role"], "ts": m["ts"],
                "snippet": _snippet(text, match.start(), match.end()),
            })
        for match in DOI_RE.finditer(text):
            doi = match.group(1).rstrip(".,;)")
            doi_hits.setdefault(doi, {
                "doi": doi, "role": m["role"], "ts": m["ts"],
                "snippet": _snippet(text, match.start(), match.end()),
            })
        for match in URL_RE.finditer(text):
            url = match.group(0).rstrip(".,;)")
            if any(skip in url for skip in SKIP_DOMAINS):
                continue
            url_hits.setdefault(url, {
                "url": url, "role": m["role"], "ts": m["ts"],
                "snippet": _snippet(text, match.start(), match.end()),
            })

    transcript = "\n\n".join(
        f"[{m['role']} @ {m['ts'][:19]}]\n{m['text']}" for m in messages
    )

    return {
        "session_file": str(jsonl),
        "resolve_note": note,
        "since_ts": since_ts,
        "message_count": len(messages),
        "arxiv": list(arxiv_hits.values()),
        "dois": list(doi_hits.values()),
        "urls": list(url_hits.values()),
        "transcript": transcript,
    }


def _snippet(text: str, start: int, end: int, window: int = 80) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return text[lo:hi].replace("\n", " ")


def mark_done(ts: str | None = None, transcript_path: str | None = None) -> dict:
    """Record the last-digest timestamp keyed by session id.

    State layout:
      {
        "sessions": {
          "<session_id>": {"last_digest_ts": "...", "transcript_path": "..."},
          ...
        },
        "last_digest_ts": "..."   # kept for backwards compat + global fallback
      }
    """
    import datetime as _dt
    now = ts or _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    jsonl, _ = resolve_transcript(transcript_path)
    state = load_state()
    sessions = state.get("sessions") or {}
    if jsonl:
        session_id = jsonl.stem
        sessions[session_id] = {
            "last_digest_ts": now,
            "transcript_path": str(jsonl),
        }
        state["sessions"] = sessions
        state["last_digest_session"] = session_id
    state["last_digest_ts"] = now  # global cursor (used when session unknown)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    return state


def cutoff_for_session(transcript_path: str | None = None) -> str:
    """Return the last_digest_ts to use for the current session.
    Prefers per-session state; falls back to global last_digest_ts."""
    state = load_state()
    jsonl, _ = resolve_transcript(transcript_path)
    if jsonl:
        session_id = jsonl.stem
        sessions = state.get("sessions") or {}
        per = sessions.get(session_id)
        if per and per.get("last_digest_ts"):
            return per["last_digest_ts"]
    return state.get("last_digest_ts", "2000-01-01T00:00:00Z")


def count_candidates(since_ts: str, transcript_path: str | None = None) -> int:
    """For hook use: return number of hard candidates since cutoff."""
    result = scan(since_ts, transcript_path=transcript_path)
    if "error" in result:
        return 0
    return len(result["arxiv"]) + len(result["dois"]) + len(result["urls"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    scan_cmd = sub.add_parser("scan", help="Scan session and emit candidates + transcript as JSON")
    scan_cmd.add_argument("--since", help="ISO timestamp override")
    scan_cmd.add_argument("--transcript", help="Explicit transcript path (skip session registry lookup)")
    scan_cmd.add_argument("--no-transcript", action="store_true", help="Omit transcript body from output")

    count_cmd = sub.add_parser("count", help="Emit just the candidate count (for hooks)")
    count_cmd.add_argument("--since", help="ISO timestamp override")
    count_cmd.add_argument("--transcript", help="Explicit transcript path")

    mark_cmd = sub.add_parser("mark-done", help="Update state file with current timestamp")
    mark_cmd.add_argument("--ts", help="ISO timestamp override (default: now UTC)")
    mark_cmd.add_argument("--transcript", help="Explicit transcript path")

    state_cmd = sub.add_parser("state", help="Print current state file")

    resolve_cmd = sub.add_parser("resolve", help="Print which session transcript would be used")
    resolve_cmd.add_argument("--transcript", help="Explicit transcript path override")

    args = ap.parse_args()

    if args.cmd == "scan":
        since = args.since or cutoff_for_session(args.transcript)
        out = scan(since, transcript_path=args.transcript)
        if args.no_transcript:
            out.pop("transcript", None)
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return

    if args.cmd == "count":
        since = args.since or cutoff_for_session(args.transcript)
        print(count_candidates(since, transcript_path=args.transcript))
        return

    if args.cmd == "mark-done":
        state = mark_done(args.ts, transcript_path=args.transcript)
        json.dump(state, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return

    if args.cmd == "state":
        json.dump(load_state(), sys.stdout, ensure_ascii=False, indent=2)
        print()
        return

    if args.cmd == "resolve":
        tp, note = resolve_transcript(args.transcript)
        json.dump({"transcript_path": str(tp) if tp else None, "note": note, "self_ppid": os.getppid()},
                  sys.stdout, ensure_ascii=False, indent=2)
        print()
        return


if __name__ == "__main__":
    main()
