#!/usr/bin/env python3
"""
Golden-set evaluation: run the example searches through the demo API in both
modes and print side-by-side top-5 lists. This is the seed of the relevance
harness — grade the columns, track the score per release.

Usage: python3 eval.py [--host http://localhost:3111]
"""

import argparse
import csv
import json
import urllib.request

def hit(host, query, mode):
    body = json.dumps({"query": query, "mode": mode, "reload": True}).encode()
    req = urllib.request.Request(f"{host}/api/search", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:3111")
    args = ap.parse_args()

    queries = [r["search_text"] for r in csv.DictReader(open("example_app_searches.csv"))]
    # add semantic queries no keyword system could answer — the demo's thesis
    queries += [
        "soft piano",
        "melancholic acoustic guitar",
        "songs about heartbreak",
        "chill instrumental to study to",
        "female vocals bossa nova",
    ]

    def rows(resp):
        out = [f"{r['song']['title'][:30]} — {r['song']['artist'][:14]}" for r in resp["results"][:5]]
        out += [f"[artist] {a['artist'][:26]} ({'; '.join(a['evidence'])[:24]})" for a in resp.get("artists", [])[:5 - len(out)]]
        return out + [""] * (5 - len(out))

    for q in queries:
        kw = hit(args.host, q, "keyword")
        hy = hit(args.host, q, "hybrid")
        plan = hy.get("plan", {})
        print(f"\n=== {q!r}  [intent: {plan.get('intent', '?')}] ===")
        if plan.get("filters"):
            print(f"    understood filters: {plan['filters']}")
        print(f"    {'KEYWORD ONLY':50s} | AI HYBRID")
        for l, r in zip(rows(kw), rows(hy)):
            print(f"    {l:50s} | {r}")
        if hy.get("note"):
            print(f"    note: {hy['note'][:120]}")

if __name__ == "__main__":
    main()
