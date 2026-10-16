"""Snapshot what was added to the Atlas, and when, from git history (for the "What changed" digest).

Reads `git log --diff-filter=A` for content records (atlas/content/<collection>/<id>.yaml) and pages
(atlas/src/pages/**.astro) and writes one entry per commit that added any of them. Run on a full clone;
a shallow clone would make everything look new.

Usage:
  python -m atlas_pipeline.digest .. ../atlas/src/data/digest.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SEP = "\x1e"
CONTENT = re.compile(r"^atlas/content/([a-z-]+)/([a-z0-9-]+)\.yaml$")
PAGE = re.compile(r"^atlas/src/pages/(.+)\.astro$")


def page_url(rel: str) -> str | None:
    """Site path for a page file; dynamic routes ([id]) are skipped."""
    if "[" in rel:
        return None
    rel = re.sub(r"(^|/)index$", "", rel)
    return "/" + (rel + "/" if rel else "")


def parse_log(text: str) -> list[dict]:
    entries = []
    for block in text.split(SEP):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        commit, date, subject = lines[0].split("\t", 2)
        records: dict[str, list[str]] = {}
        pages = []
        for path in lines[1:]:
            if m := CONTENT.match(path):
                records.setdefault(m.group(1), []).append(m.group(2))
            elif (m := PAGE.match(path)) and (u := page_url(m.group(1))):
                pages.append(u)
        if records or pages:
            entries.append({"commit": commit[:10], "date": date[:10], "subject": subject, "records": records, "pages": sorted(pages)})
    return entries


def git_log(repo: Path) -> str:
    fmt = f"{SEP}%H%x09%aI%x09%s"
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--diff-filter=A", "--name-only", f"--format={fmt}", "--", "atlas/content", "atlas/src/pages"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return out.stdout


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    repo = Path(argv[1])
    shallow = subprocess.run(["git", "-C", str(repo), "rev-parse", "--is-shallow-repository"], capture_output=True, text=True).stdout.strip()
    if shallow == "true":
        print("refusing to run on a shallow clone: every file would look newly added", file=sys.stderr)
        return 1
    entries = parse_log(git_log(repo))
    out = Path(argv[2])
    out.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(entries)} commits with additions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
