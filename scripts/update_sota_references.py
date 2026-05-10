#!/usr/bin/env python3
"""
Fetch recent arXiv papers for repo-relevant keywords and write a markdown report.

Intended to be run from CI on a schedule. Does NOT push by itself; the workflow
commits and pushes the generated candidate report to the current branch.

Usage:
  python scripts/update_sota_references.py --output docs/SOTA_AUTOMATION_UPDATES.md

"""
import argparse
import datetime
import requests
import xml.etree.ElementTree as ET
import os
import re


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="docs/SOTA_AUTOMATION_UPDATES.md")
    p.add_argument("--guide", default="docs/SOTA_AUTOMATION_GUIDE.md")
    p.add_argument("--max-per-query", type=int, default=5)
    args = p.parse_args()

    existing = load_existing_titles(args.guide)

    found = []
    for q in QUERIES:
        try:
            items = query_arxiv(q, max_results=args.max_per_query)
        except Exception as e:
            print(f"query failed for {q}: {e}")
            continue
        for it in items:
            key = it["title"].lower()
            if key in existing:
                continue
            found.append({"query": q, **it})

    now = datetime.datetime.utcnow().isoformat()[:19] + "Z"
    out_lines = ["# SOTA Automation - Candidate Updates", "", f"Generated: {now}", "", "## New papers found", ""]
    if not found:
        out_lines.append("No new papers found for configured queries.")
    else:
        for it in found:
            out_lines.append(f"### {it['title']}")
            out_lines.append("")
            out_lines.append(f"**Published:** {it['published']}")
            out_lines.append("")
            out_lines.append(f"**Authors:** {', '.join(it['authors'])}")
            out_lines.append("")
            out_lines.append(f"{it['summary']}")
            out_lines.append("")
            out_lines.append(f"[arXiv]({it['link']})")
            out_lines.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    open(args.output, "w", encoding="utf-8").write("\n".join(out_lines))
    print(f"Wrote report to {args.output} (found {len(found)} items)")


if __name__ == "__main__":
    main()
