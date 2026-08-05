#!/usr/bin/env python3
"""
Djaminn discovery-test search prototype.

Usage:
    python3 search.py "piano"            # single query
    python3 search.py --demo             # run every query in example_app_searches.csv
    python3 search.py "artists near me" --region "LATAM"

Design (deliberately v1, no ML deps):
  query -> normalize -> intent route -> concept expansion (instrument/genre lexicon)
        -> field-weighted keyword scoring over artist & song docs
        -> blend with engagement/recency priors -> explainable ranked results

The point of the prototype is judgement, not infrastructure: it shows which of
the example searches the raw export can and cannot answer, and makes every
ranking decision inspectable ("why" column).
"""

import csv
import re
import sys
import math
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone

import openpyxl

XLSX = "DJaminn_Top_Artists_Songs_original_candidate_data.xlsx"
SEARCHES_CSV = "example_app_searches.csv"

# The "Djaminn Artists" rank-1 profile carries 38 genres and reads like a
# house/aggregate account; left in results but damped so it doesn't top
# every genre search.
HOUSE_ACCOUNTS = {"Djaminn Artists"}
HOUSE_DAMP = 0.4


# ---------------------------------------------------------------- normalization

def fold(text):
    """Lowercase + strip diacritics so 'Segré'/'João' match 'segre'/'joao'."""
    text = unicodedata.normalize("NFD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokens(text):
    return re.findall(r"[^\W_]+", fold(text), re.UNICODE)


# Raw genre tags are inconsistent ("Singer Songwriters" vs "Singer/Songwriter").
GENRE_ALIASES = {
    "singer songwriters": "singer-songwriter",
    "singer/songwriter": "singer-songwriter",
    "rap/hip-hop": "hip-hop",
    "pop rap": "pop-rap",
    "electronic/dance": "electronic",
    "folk/country": "folk-country",
    "funk/soul": "funk-soul",
}


def norm_genre(tag):
    t = fold(tag).strip()
    return GENRE_ALIASES.get(t, t)


# ------------------------------------------------------- concept lexicon
# Maps what a musician *types* to the fields/vocabulary this dataset actually
# has: instrument/role words to title/name evidence, genre words to tags.

CONCEPTS = {
    "piano": {
        "title": {"piano", "keys", "keyboard", "rhodes"},
        "name": {"piano"},
        "genres": set(),
    },
    "drummer": {
        "title": {"drum", "drums", "drumbeat", "percussion"},
        "name": {"drum"},
        "genres": set(),
    },
    "drum": "drummer",  # alias
    "drums": "drummer",
    "guitar": {
        "title": {"guitar", "acoustic", "riff"},
        "name": {"guitar"},
        "genres": set(),
    },
    "guitarist": "guitar",
    "vocalist": {
        "title": {"vocal", "vocals", "voice", "sing", "singer", "acapella"},
        "name": {"singer"},
        "genres": {"singer-songwriter"},
        # a cover on this platform is near-always a sung performance;
        # weak evidence, weighted down below
        "weak_title": {"cover"},
    },
    "singer": "vocalist",
    "vocals": "vocalist",
    "hip hop": {
        "title": {"rap", "freestyle", "trap", "flow"},
        "name": set(),
        "genres": {"hip-hop", "pop-rap", "trap", "gangsta"},
    },
    "hiphop": "hip hop",
    "rap": "hip hop",
    "latin": {
        "title": {"samba", "bossa", "salsa", "cumbia", "tango", "ipanema"},
        "name": set(),
        "genres": {"latin", "latin jazz"},
    },
    "ambient": {
        "title": {"ambient", "chill", "atmosphere", "soundscape"},
        "name": set(),
        "genres": {"ambient"},
    },
    "electronic": {
        "title": {"remix", "mix", "edit", "house", "techno"},
        "name": set(),
        "genres": {"electronic", "house", "deep house", "tech house",
                   "tech", "minimal", "disco", "breakbeat dnb"},
    },
    "jazz": {
        "title": {"jazz", "swing", "bebop"},
        "name": set(),
        "genres": {"jazz", "smooth jazz", "contemporary jazz", "latin jazz",
                   "bebop", "big band swing", "fusion"},
    },
    "rock": {
        "title": {"rock"},
        "name": set(),
        "genres": {"rock", "hard rock", "blues rock", "folk rock",
                   "rock and roll", "psychedelic progressive rock",
                   "pop soft rock", "indie alternative"},
    },
}


def resolve_concept(key):
    c = CONCEPTS.get(key)
    while isinstance(c, str):
        c = CONCEPTS.get(c)
    return c


# ---------------------------------------------------------------- data loading

def _sheet_dicts(wb, name):
    rows = wb[name].iter_rows(values_only=True)
    header = next(rows)
    return [dict(zip(header, r)) for r in rows if any(c is not None for c in r)]


def parse_date(s):
    if not s:
        return None
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def load():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    users = _sheet_dicts(wb, "User")
    songs = _sheet_dicts(wb, "Songs")

    by_id = {u["USER_ID"]: u for u in users}
    for u in users:
        u["_genres"] = [norm_genre(g) for g in str(u["GENRES"] or "").split(",") if g.strip()]
        u["_name_tokens"] = tokens(u["ARTIST_NAME"])
        u["_songs"] = []

    for s in songs:
        s["_date"] = parse_date(s["UPLOAD_DATE"])
        s["_title_tokens"] = tokens(s["SONG_TITLE"])
        s["_artist"] = by_id.get(s["USER_ID"])
        if s["_artist"]:
            s["_artist"]["_songs"].append(s)

    # Derived activity. NOTE: export caps at 20 songs/artist, so first-seen
    # date is a *biased* proxy for "joined recently" — flagged in output.
    anchor = max(s["_date"] for s in songs)
    for u in users:
        ds = [s["_date"] for s in u["_songs"]]
        u["_last_upload"] = max(ds) if ds else None
        u["_first_seen"] = min(ds) if ds else None
        u["_days_idle"] = (anchor - u["_last_upload"]).days if ds else None
        u["_capped"] = u["SONG_COUNT"] and u["SONG_COUNT"] > len(u["_songs"])

    max_apb = max(u["PLAYBACK_COUNT"] or 0 for u in users) or 1
    for u in users:
        pop = math.log1p(u["PLAYBACK_COUNT"] or 0) / math.log1p(max_apb)
        rec = math.exp(-(u["_days_idle"] or 999) / 90)
        u["_prior"] = 0.6 * pop + 0.4 * rec
        if u["ARTIST_NAME"] in HOUSE_ACCOUNTS:
            u["_prior"] *= HOUSE_DAMP

    max_spb = max(s["PLAYBACK_COUNT"] or 0 for s in songs) or 1
    for s in songs:
        pop = math.log1p(s["PLAYBACK_COUNT"] or 0) / math.log1p(max_spb)
        rec = math.exp(-(anchor - s["_date"]).days / 120) if s["_date"] else 0
        s["_prior"] = 0.7 * pop + 0.3 * rec
        if s["ARTIST_NAME"] in HOUSE_ACCOUNTS:
            s["_prior"] *= HOUSE_DAMP

    return users, songs, anchor


# ---------------------------------------------------------------- scoring

W = {
    "artist_name_exact": 3.5, "artist_name_sub": 2.5, "artist_genre": 2.5,
    "artist_region": 1.0, "title_exact": 2.5, "title_phrase": 4.0,
    "title_weak": 0.8, "song_artist_name": 1.5, "song_genre": 1.2,
    "prior_blend": 0.5,
}


def expand(query):
    """Query -> (strong tokens, weak tokens, genre targets, matched concepts)."""
    q = fold(query).strip()
    qtok = tokens(query)
    title_t, name_t, weak_t, genre_t, hits = set(qtok), set(qtok), set(), set(), []

    for key in [q] + qtok:  # try whole phrase first, then each word
        c = resolve_concept(key)
        if c:
            hits.append(key)
            title_t |= c["title"]
            name_t |= c["name"]
            genre_t |= c["genres"]
            weak_t |= c.get("weak_title", set())
    # raw query words might themselves be genre tags ("ambient", "latin")
    genre_t |= {norm_genre(t) for t in qtok}
    if q:
        genre_t.add(norm_genre(q))
    return title_t, name_t, weak_t, genre_t, hits


def score_artist(u, title_t, name_t, genre_t):
    score, why = 0.0, []
    ntoks = set(u["_name_tokens"])
    for t in name_t | title_t:
        if t in ntoks:
            score += W["artist_name_exact"]; why.append(f"name:{t}")
        elif len(t) >= 4 and any(t in nt for nt in ntoks):
            score += W["artist_name_sub"]; why.append(f"name~{t}")
    g = genre_t & set(u["_genres"])
    if g:
        score += W["artist_genre"] * min(len(g), 3)
        why.append("genre:" + ",".join(sorted(g)[:3]))
    # song titles are evidence about the artist too (a profile with 5 drum
    # jams is a drummer even though no field says so)
    hits = sum(1 for s in u["_songs"] if set(s["_title_tokens"]) & title_t)
    if hits:
        score += min(hits, 4) * 1.0
        why.append(f"songs:{hits} matching titles")
    if score:
        score *= 1 + W["prior_blend"] * u["_prior"]
    return score, why


def score_song(s, title_t, weak_t, name_t, genre_t):
    score, why = 0.0, []
    ttoks = set(s["_title_tokens"])
    strong = title_t & ttoks
    if strong:
        score += W["title_exact"] * len(strong)
        why.append("title:" + ",".join(sorted(strong)))
    weak = weak_t & ttoks
    if weak and not strong:
        score += W["title_weak"]
        why.append("title(weak):" + ",".join(sorted(weak)))
    a = s["_artist"]
    if a:
        if set(a["_name_tokens"]) & (name_t | title_t):
            score += W["song_artist_name"]; why.append("artist-name")
        g = genre_t & set(a["_genres"])
        if g and score:  # inherited genre only reinforces, never matches alone
            score += W["song_genre"] * min(len(g), 2)
            why.append("artist-genre:" + ",".join(sorted(g)[:2]))
        elif g and not score:
            score += W["song_genre"] * 0.6 * min(len(g), 2)
            why.append("artist-genre-only:" + ",".join(sorted(g)[:2]))
    if score:
        score *= 1 + W["prior_blend"] * s["_prior"]
    return score, why


# ---------------------------------------------------------------- intents

def intent_of(query):
    q = fold(query).strip()
    if re.search(r"\bnew\b.*\bartist|\bartist.*\bnew\b", q):
        return "new_artists"
    if "popular" in q or "top song" in q or "trending" in q:
        return "popular_songs"
    if "collab" in q:
        return "collaborators"
    if "near me" in q or "nearby" in q or "near by" in q:
        return "near_me"
    if "remix" in q and ("for" in q or "music" in q or "to" in q):
        return "remixable"
    return "keyword"


def run_new_artists(users, anchor, k):
    ranked = sorted((u for u in users if u["_first_seen"]),
                    key=lambda u: u["_first_seen"], reverse=True)
    out = []
    for u in ranked[:k]:
        why = [f"first upload in export {u['_first_seen'].date()}"]
        if u["_capped"]:
            why.append("PROXY-BIASED: export capped at 20 songs, real join date unknown")
        out.append(("artist", u["ARTIST_NAME"], 0, why, u))
    return out, ("No signup/join date in the export — 'new' is proxied by earliest "
                 "upload present, which is biased for the 15 capped artists.")


def run_popular_songs(songs, k):
    ranked = sorted(songs, key=lambda s: (s["PLAYBACK_COUNT"] or 0), reverse=True)
    seen, out = defaultdict(int), []
    for s in ranked:  # cap 2 per artist so one uploader doesn't own the list
        if seen[s["ARTIST_NAME"]] >= 2:
            continue
        seen[s["ARTIST_NAME"]] += 1
        out.append(("song", f"{s['SONG_TITLE']} — {s['ARTIST_NAME']}", 0,
                    [f"{int(s['PLAYBACK_COUNT'] or 0)} plays",
                     f"uploaded {s['_date'].date()}"], s))
        if len(out) == k:
            break
    return out, ("Raw lifetime playback favours old uploads; a production version "
                 "should use windowed plays (e.g. last 30 days), which this export "
                 "doesn't contain.")


def run_collaborators(users, k):
    scored = []
    for u in users:
        if u["ARTIST_NAME"] in HOUSE_ACCOUNTS or not u["_songs"]:
            continue
        active = math.exp(-(u["_days_idle"] or 999) / 45)
        social = math.log1p(u["FOLLOW_COUNT"] or 0) / math.log(101)
        wanted = math.log1p(u["FOLLOWER_COUNT"] or 0) / math.log(3001)
        s = 0.5 * active + 0.3 * min(social, 1) + 0.2 * min(wanted, 1)
        why = [f"last active {u['_days_idle']}d before export",
               f"follows {int(u['FOLLOW_COUNT'] or 0)}",
               f"{int(u['FOLLOWER_COUNT'] or 0)} followers"]
        scored.append(("artist", u["ARTIST_NAME"], s, why, u))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:k], ("No collaboration/project/messaging data in the export — this is "
                        "an 'active + socially engaged' proxy, not real collab intent.")


def run_near_me(users, region, k):
    if not region:
        regions = defaultdict(int)
        for u in users:
            regions[u["REGION"]] += 1
        rows = [("region", f"{r}  ({n} artists)", 0, [], None)
                for r, n in sorted(regions.items(), key=lambda x: -x[1])]
        return rows[:k], ("No city/geo data — REGION is 6 coarse buckets. True 'near me' "
                          "needs user location + geo on profiles. Pass --region to filter.")
    rt = fold(region)
    hits = [u for u in users if rt in fold(u["REGION"] or "")]
    hits.sort(key=lambda u: u["_prior"], reverse=True)
    return ([("artist", u["ARTIST_NAME"], u["_prior"], [u["REGION"]], u) for u in hits[:k]],
            "Region-bucket match only — not actual proximity.")


def run_remixable(songs, k):
    inst = {"beat", "jam", "loop", "groove", "instrumental", "drum", "riff",
            "solo", "random", "track", "remix", "mix"}
    scored = []
    for s in songs:
        tt = set(s["_title_tokens"])
        hits = tt & inst
        if not hits:
            continue
        pen_cover = 0.5 if "cover" in tt else 1.0  # covers = rights risk for remixing
        sc = (1 + len(hits)) * pen_cover * (1 + s["_prior"])
        why = ["title:" + ",".join(sorted(hits))]
        if pen_cover < 1:
            why.append("cover (rights risk, downranked)")
        scored.append(("song", f"{s['SONG_TITLE']} — {s['ARTIST_NAME']}", sc, why, s))
    scored.sort(key=lambda x: x[2], reverse=True)
    return diversify(scored, k), (
        "Title keywords only — no stems/instrumental flag exists, though "
        "every song has a MIXDOWN_AUDIO file the platform could analyse.")


def diversify(results, k, max_per_artist=2):
    """Cap songs per artist so one uploader's near-identical takes don't own
    the page; artists themselves are never capped."""
    seen, out = defaultdict(int), []
    for r in results:
        if r[0] == "song":
            artist = r[4]["ARTIST_NAME"]
            if seen[artist] >= max_per_artist:
                continue
            seen[artist] += 1
        out.append(r)
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------- driver

def run_query(query, users, songs, anchor, k=5, region=None):
    intent = intent_of(query)
    if intent == "new_artists":
        return intent, *run_new_artists(users, anchor, k)
    if intent == "popular_songs":
        return intent, *run_popular_songs(songs, k)
    if intent == "collaborators":
        return intent, *run_collaborators(users, k)
    if intent == "near_me":
        return intent, *run_near_me(users, region, k)
    if intent == "remixable":
        return intent, *run_remixable(songs, k)

    title_t, name_t, weak_t, genre_t, hits = expand(query)
    results = []
    for u in users:
        sc, why = score_artist(u, title_t, name_t, genre_t)
        if sc > 0:
            results.append(("artist", u["ARTIST_NAME"], sc, why, u))
    for s in songs:
        sc, why = score_song(s, title_t, weak_t, name_t, genre_t)
        if sc > 0:
            results.append(("song", f"{s['SONG_TITLE']} — {s['ARTIST_NAME']}", sc, why, s))
    results.sort(key=lambda x: x[2], reverse=True)
    note = f"expanded via concepts: {', '.join(sorted(set(hits)))}" if hits else \
        "no concept match — plain keyword scoring"
    return intent, diversify(results, k), note


def show(query, intent, results, note, k=5):
    print(f"\n=== \"{query}\"  [intent: {intent}] ===")
    if not results:
        print("  (no results)")
    for i, (kind, label, sc, why, _) in enumerate(results[:k], 1):
        s = f" ({sc:.2f})" if sc else ""
        print(f"  {i}. [{kind}] {label}{s}")
        if why:
            print(f"       why: {'; '.join(why)}")
    print(f"  note: {note}")


def main():
    args = [a for a in sys.argv[1:]]
    region = None
    if "--region" in args:
        i = args.index("--region")
        region = args[i + 1]
        del args[i:i + 2]

    users, songs, anchor = load()
    print(f"Loaded {len(users)} artists, {len(songs)} songs "
          f"(export anchor date {anchor.date()})")

    if "--demo" in args:
        with open(SEARCHES_CSV) as f:
            queries = [r["search_text"] for r in csv.DictReader(f)]
        for q in queries:
            intent, results, note = run_query(q, users, songs, anchor, region=region)
            show(q, intent, results, note)
    elif args:
        q = " ".join(args)
        intent, results, note = run_query(q, users, songs, anchor, k=10, region=region)
        show(q, intent, results, note, k=10)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
