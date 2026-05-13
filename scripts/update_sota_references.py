#!/usr/bin/env python3
"""
Fetch recent arXiv papers for repo-relevant keywords and refresh the guide.

Intended to be run from CI on a schedule. Does NOT push by itself; the workflow
commits and pushes the updated guide to the current branch.

Usage:
  python scripts/update_sota_references.py --output docs/SOTA_AUTOMATION_GUIDE.md

"""
import argparse
import datetime
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

QUERIES = [
    "out-of-distribution detection",
    "energy based ood",
    "mahalanobis ood",
    "logitnorm",
    "selective prediction",
    "segment anything",
    "sam segmentation",
    "bioclip",
    "router calibration",
    "conformal prediction",
]


ARXIV_API = "http://export.arxiv.org/api/query"
CANDIDATE_SECTION_BEGIN = "<!-- BEGIN SOTA AUTOMATION CANDIDATES -->"
CANDIDATE_SECTION_END = "<!-- END SOTA AUTOMATION CANDIDATES -->"


def query_arxiv(q, max_results=5):
    params = {
        "search_query": f"all:{q}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        link = entry.find("atom:id", ns).text.strip()
        published = entry.find("atom:published", ns).text.strip()
        authors = [a.find("atom:name", ns).text.strip() for a in entry.findall("atom:author", ns)]
        items.append({"title": title, "summary": summary, "link": link, "published": published, "authors": authors})
    return items


def load_existing_titles(guide_path):
    if not os.path.exists(guide_path):
        return set()
    text = open(guide_path, "r", encoding="utf-8").read()
    titles = set()
    # crude heuristic: find lines in Literature Anchors tables or bullets that look like titles
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 200 and (line.startswith("|") or line.startswith("- ") or line.startswith("* ")):
            # remove table pipes
            cleaned = re.sub(r"[|*\-]", "", line).strip()
            if cleaned and len(cleaned) > 10:
                titles.add(cleaned.lower())
    return titles


def summarize_query_error(error):
    error_text = str(error)
    if "WinError 10013" in error_text:
        return "network access blocked by local permissions"
    if "timed out" in error_text.lower():
        return "query timed out"
    return error_text.splitlines()[0][:200]


def render_candidate_section(found, generated_at, query_errors=None):
    query_errors = query_errors or []
    out_lines = [
        CANDIDATE_SECTION_BEGIN,
        "#### Latest Automated Candidate Scan",
        "",
        f"Generated: `{generated_at}`",
        "",
        "These are machine-collected literature candidates for human review. They are not accepted repo guidance until a maintainer promotes them into the relevant Literature Anchors table above.",
        "",
    ]
    if query_errors:
        out_lines.append("Candidate scan could not query all configured sources:")
        out_lines.append("")
        for query, error in query_errors:
            out_lines.append(f"- `{query}`: {summarize_query_error(error)}")
        out_lines.append("")

    if not found and not query_errors:
        out_lines.append("No new papers found for configured queries.")
    elif found:
        for it in found:
            out_lines.append(f"##### {it['title']}")
            out_lines.append("")
            out_lines.append(f"- Query: `{it['query']}`")
            out_lines.append(f"- Published: `{it['published']}`")
            out_lines.append(f"- Authors: {', '.join(it['authors'])}")
            out_lines.append(f"- Link: {it['link']}")
            out_lines.append(f"- Review note: {it['summary']}")
            out_lines.append("")
    out_lines.extend(["", CANDIDATE_SECTION_END])
    return "\n".join(out_lines)


def update_guide_with_candidate_section(guide_text, candidate_section):
    if CANDIDATE_SECTION_BEGIN in guide_text and CANDIDATE_SECTION_END in guide_text:
        pattern = re.compile(
            rf"{re.escape(CANDIDATE_SECTION_BEGIN)}.*?{re.escape(CANDIDATE_SECTION_END)}",
            flags=re.DOTALL,
        )
        return pattern.sub(candidate_section, guide_text)

    phase_2_heading = "### Phase 2 Checklist"
    if phase_2_heading in guide_text:
        return guide_text.replace(phase_2_heading, f"{candidate_section}\n\n{phase_2_heading}", 1)

    return guide_text.rstrip() + "\n\n" + candidate_section + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="docs/SOTA_AUTOMATION_GUIDE.md")
    p.add_argument("--guide", default="docs/SOTA_AUTOMATION_GUIDE.md")
    p.add_argument("--max-per-query", type=int, default=5)
    args = p.parse_args()

    existing = load_existing_titles(args.guide)

    found = []
    query_errors = []
    for q in QUERIES:
        try:
            items = query_arxiv(q, max_results=args.max_per_query)
        except Exception as e:
            print(f"query failed for {q}: {e}")
            query_errors.append((q, str(e)))
            continue
        for it in items:
            key = it["title"].lower()
            if key in existing:
                continue
            found.append({"query": q, **it})

    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    output_path = Path(args.output)
    guide_path = Path(args.guide)
    candidate_section = render_candidate_section(found, now, query_errors=query_errors)

    if output_path == guide_path:
        guide_text = guide_path.read_text(encoding="utf-8")
        output_text = update_guide_with_candidate_section(guide_text, candidate_section)
    else:
        output_text = candidate_section

    os.makedirs(output_path.parent, exist_ok=True)
    output_path.write_text(output_text.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote SOTA candidate scan to {args.output} (found {len(found)} items)")


if __name__ == "__main__":
    main()
