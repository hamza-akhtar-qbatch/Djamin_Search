#!/usr/bin/env python3
"""
Add a local reference track to the search index as a test fixture.

Runs the exact same Gemini audio analysis + embedding as enrich.py, but writes
to test_tracks.jsonl (merged into the index at compile time, flagged
is_test_fixture). Lets us validate the pipeline against tracks whose ground
truth everyone knows.

Usage:
    python3 add_track.py audio_cache/test_despacito.mp3 \
        --title "Despacito" --artist "Luis Fonsi ft. Daddy Yankee" \
        --serve-copy web/public/test_audio
"""

import argparse
import json
import os
import shutil

from enrich import analyze_audio, embed, profile_text

TEST_FILE = "test_tracks.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp3")
    ap.add_argument("--title", required=True)
    ap.add_argument("--artist", required=True)
    ap.add_argument("--serve-copy", default="web/public/test_audio",
                    help="dir under web/public to copy the mp3 for local playback")
    args = ap.parse_args()

    song_id = "test_" + os.path.splitext(os.path.basename(args.mp3))[0]

    os.makedirs(args.serve_copy, exist_ok=True)
    served = os.path.join(args.serve_copy, song_id + ".mp3")
    shutil.copyfile(args.mp3, served)
    audio_url = "/" + os.path.relpath(served, "web/public")

    print(f"Analyzing {args.mp3} ...")
    analysis = analyze_audio(args.mp3)
    song_stub = {"SONG_TITLE": args.title, "ARTIST_NAME": args.artist,
                 "artist_genres": []}
    text = profile_text(song_stub, analysis)
    vector = embed(text)

    rec = {
        "song_id": song_id,
        "title": args.title,
        "artist": args.artist,
        "artist_id": song_id + "_artist",
        "audio_url": audio_url,
        "upload_date": "2026-06-01T00:00:00.000Z",
        "playback_count": 0,
        "duration_seconds": 0,
        "region": "test fixture",
        "artist_genres": [],
        "artist_rating": None,
        "artist_followers": None,
        "is_test_fixture": True,
        "analysis": analysis,
        "profile_text": text,
        "embedding": vector,
    }
    with open(TEST_FILE, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    a = analysis
    print(f"Added {args.title!r}: genres={a['genres']}, moods={a['moods'][:4]}, "
          f"bpm={a['bpm_estimate']}, lang={a['language']}")
    print(f"Run 'python3 enrich.py --compile' to rebuild the index.")


if __name__ == "__main__":
    main()
