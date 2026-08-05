#!/usr/bin/env python3
"""
Second-stage enrichment: cross-lingual meaning + sentiment.

For every enriched song, a cheap TEXT-ONLY model call produces:
  - lyrics_translation_en : full English translation of the transcription
  - themes                : what the song is about (lowercase phrases)
  - sentiment             : one emotional label

Then the profile text is rebuilt and re-embedded so an English query matches
a Spanish/Portuguese/Hindi/Thai song by meaning. Resume-safe: records that
already carry `themes` are skipped; files rewritten atomically at the end.

Usage: /usr/bin/python3 relingua.py
"""

import json
import os

from enrich import post_json, embed, profile_text, API, KEY

TEXT_MODEL = "gemini-3.1-flash-lite"  # text quota is separate from audio models
FILES = ["enriched.jsonl", "test_tracks.jsonl"]

SCHEMA = {
    "type": "object",
    "properties": {
        "lyrics_translation_en": {"type": "string", "description": "Full English translation of the lyrics. Empty string if instrumental or already English."},
        "themes": {"type": "array", "items": {"type": "string"}, "description": "2-5 lowercase phrases for what the song is about, e.g. 'heartbreak', 'missing someone', 'faith', 'partying', 'social critique'"},
        "sentiment": {"type": "string", "enum": ["joyful", "sad", "romantic", "heartbroken", "angry", "hopeful", "nostalgic", "energetic", "calm", "mixed"]},
    },
    "required": ["lyrics_translation_en", "themes", "sentiment"],
}


def analyze_text(rec):
    a = rec["analysis"]
    prompt = (
        "You are enriching a music search index for cross-lingual meaning search.\n"
        f"Song: {str(rec['title'])!r} by {rec['artist']}\n"
        f"Language: {a.get('language')}\n"
        f"Description: {a.get('description', '')}\n"
        f"Lyrics gist: {a.get('lyrics_gist', '')}\n"
        f"Transcription (original language):\n{(a.get('transcription') or '')[:2500]}\n\n"
        "Produce: a faithful full English translation of the lyrics (empty if instrumental "
        "or already fully English), 2-5 theme phrases describing what the song is about, "
        "and one overall sentiment label. Base everything only on the material above."
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
            "temperature": 0,
        },
    }
    r = post_json(f"{API}/{TEXT_MODEL}:generateContent?key={KEY}", body, tries=5)
    return json.loads(r["candidates"][0]["content"]["parts"][0]["text"])


def main():
    for path in FILES:
        if not os.path.exists(path):
            continue
        records = [json.loads(l) for l in open(path) if l.strip()]
        for rec in records:  # old records may carry numeric titles
            if "title" in rec:
                rec["title"] = str(rec["title"])
        todo = [r for r in records if "error" not in r and not r["analysis"].get("themes")]
        print(f"{path}: {len(records)} records, {len(todo)} need the multilingual pass")

        def persist():
            tmp = path + ".v2"
            with open(tmp, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, path)

        done = 0
        for rec in records:
            if "error" in rec or rec["analysis"].get("themes"):
                continue
            title = str(rec["title"])[:40]
            try:
                extra = analyze_text(rec)
                rec["analysis"].update(extra)
                stub = {"SONG_TITLE": rec["title"], "ARTIST_NAME": rec["artist"],
                        "artist_genres": rec.get("artist_genres", [])}
                rec["profile_text"] = profile_text(stub, rec["analysis"])
                rec["embedding"] = embed(rec["profile_text"])
                done += 1
                persist()  # crash-safe: progress lands on disk every record
                lang = rec["analysis"].get("language")
                tr = "translated" if extra["lyrics_translation_en"] else "no-translation"
                print(f"  [{done}/{len(todo)}] OK {title!r} ({lang}) "
                      f"{tr} · themes={extra['themes'][:3]} · {extra['sentiment']}", flush=True)
            except Exception as e:
                print(f"  FAIL {title!r}: {str(e)[:120]}", flush=True)
        persist()
        print(f"{path}: updated in place ({done} enriched this run)")


if __name__ == "__main__":
    main()
