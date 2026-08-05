"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface Analysis {
  description: string;
  transcription: string;
  lyrics_gist: string;
  instruments: string[];
  moods: string[];
  genres: string[];
  energy: string;
  has_vocals: boolean;
  vocal_type: string;
  vocal_delivery: string;
  language: string;
  bpm_estimate: number;
  tempo_feel: string;
  key_guess: string;
  production: string;
  good_for_remixing: boolean;
  use_cases: string[];
  sounds_like: string[];
  lyrics_translation_en?: string;
  themes?: string[];
  sentiment?: string;
}

interface SongOut {
  song_id: string;
  title: string;
  artist: string;
  audio_url: string;
  upload_date: string;
  playback_count: number;
  duration_seconds: number;
  region: string;
  artist_genres: string[];
  analysis: Analysis;
}

interface Result {
  song: SongOut;
  score: number;
  semanticScore: number | null;
  keywordTerms: string[];
  matchedFilters: string[];
  why: string;
}

interface ArtistOut {
  artist: string;
  artist_id: string;
  songCount: number;
  genres: string[];
  topMoods: string[];
  topInstruments: string[];
  evidence: string[];
}

interface ApiResponse {
  plan: {
    semantic_query: string;
    keywords: string[];
    filters: Record<string, string | boolean>;
    result_focus: string;
  };
  results: Result[];
  artists: ArtistOut[];
  note?: string;
  timings: { planMs: number; retrievalMs: number; totalMs: number };
  error?: string;
}

const EXAMPLES = [
  "songs about heartbreak",
  "reggaeton party hit",
  "drummer",
  "female vocals bossa nova",
  "chill instrumental to study to",
  "music for remixing",
  "angry songs",
  "artists in latin america",
];

const LOADING_STAGES = [
  "Understanding your query…",
  "Searching by meaning…",
  "Ranking results…",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"hybrid" | "keyword">("hybrid");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [stage, setStage] = useState(0);
  const [intros, setIntros] = useState<Record<string, string>>({});
  const [introLoading, setIntroLoading] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const draftIntro = useCallback(
    async (artistId: string) => {
      setIntroLoading(artistId);
      try {
        const res = await fetch("/api/intro", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ artist_id: artistId, query }),
        });
        const json = await res.json();
        if (res.ok && json.message) {
          setIntros((m) => ({ ...m, [artistId]: json.message }));
        }
      } finally {
        setIntroLoading(null);
      }
    },
    [query]
  );

  useEffect(() => {
    if (!loading) {
      setStage(0);
      return;
    }
    const t = setInterval(() => setStage((s) => Math.min(s + 1, LOADING_STAGES.length - 1)), 900);
    return () => clearInterval(t);
  }, [loading]);

  const runSearch = useCallback(
    async (q: string, m: "hybrid" | "keyword" = mode) => {
      if (!q.trim()) return;
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, mode: m }),
          signal: ctrl.signal,
        });
        const json: ApiResponse = await res.json();
        if (!res.ok) throw new Error(json.error ?? `HTTP ${res.status}`);
        setData(json);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(e instanceof Error ? e.message : "search failed");
      } finally {
        setLoading(false);
      }
    },
    [mode]
  );

  const fmtDur = (s: number) =>
    `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-5xl px-4 py-10">
        <header className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">
            Djaminn <span className="text-emerald-400">AI Discovery</span>
          </h1>
          <p className="mt-1 text-sm text-zinc-400">
            Search by meaning, mood, and sound — powered by AI analysis of the
            actual audio, not just titles and tags.
          </p>
        </header>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            runSearch(query);
          }}
          className="flex gap-2"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder='Try "soft piano" or "songs about heartbreak"…'
            className="flex-1 rounded-xl border border-zinc-800 bg-zinc-900 px-4 py-3 text-base outline-none placeholder:text-zinc-500 focus:border-emerald-500"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded-xl bg-emerald-500 px-6 py-3 font-semibold text-zinc-950 transition hover:bg-emerald-400 disabled:opacity-50"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </form>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => {
                setQuery(ex);
                runSearch(ex);
              }}
              className="rounded-full border border-zinc-800 bg-zinc-900 px-3 py-1 text-xs text-zinc-300 transition hover:border-emerald-500 hover:text-emerald-300"
            >
              {ex}
            </button>
          ))}
         
        </div>

        {error && (
          <div className="mt-6 rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {data && !error && (
          <div className="mt-8 space-y-8">
            {data.note && (
              <div className="rounded-xl border border-amber-900/60 bg-amber-950/30 p-4 text-xs leading-relaxed text-amber-200/90">
                {data.note}
              </div>
            )}

            {mode === "hybrid" && (
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-xs text-zinc-400">
                <span className="font-semibold text-zinc-300">
                  Understood as:
                </span>{" "}
                {data.plan.semantic_query}
                {Object.keys(data.plan.filters).length > 0 && (
                  <>
                    {" · "}
                    <span className="font-semibold text-zinc-300">
                      filters:
                    </span>{" "}
                    {Object.entries(data.plan.filters)
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ")}
                  </>
                )}
                {" · "}
                <span className="text-zinc-500">
                  {data.timings.totalMs}ms ({data.timings.planMs}ms
                  understanding)
                </span>
              </div>
            )}

            {data.artists.length > 0 && (
              <section>
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-400">
                  Related artists
                </h2>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {data.artists.map((a) => (
                    <div
                      key={a.artist_id}
                      className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4"
                    >
                      <div className="font-semibold">{a.artist}</div>
                      <div className="mt-1 text-xs text-zinc-400">
                        {a.songCount} matching track
                        {a.songCount > 1 ? "s" : ""}
                        {a.topInstruments.length > 0 &&
                          ` · ${a.topInstruments.slice(0, 3).join(", ")}`}
                      </div>
                      {a.topMoods.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {a.topMoods.map((m) => (
                            <span
                              key={m}
                              className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300"
                            >
                              {m}
                            </span>
                          ))}
                        </div>
                      )}
                      <div className="mt-2 truncate text-[11px] text-zinc-500">
                        e.g. {a.evidence.join(" · ")}
                      </div>
                    
                      {intros[a.artist_id] && (
                        <div className="mt-2 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 text-[12px] leading-relaxed text-zinc-300">
                          {intros[a.artist_id]}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {(data.results.length > 0 || data.artists.length === 0) && (
            <section>
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-400">
                Songs
              </h2>
              <div className="space-y-3">
                {data.results.length === 0 && (
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-6 text-sm text-zinc-400">
                    No results — try a broader phrase.
                  </div>
                )}
                {data.results.map((r) => {
                  const a = r.song.analysis;
                  const isOpen = expanded === r.song.song_id;
                  return (
                    <div
                      key={r.song.song_id}
                      className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="truncate font-semibold">
                            {r.song.title}
                          </div>
                          <div className="text-sm text-zinc-400">
                            {r.song.artist} · {fmtDur(r.song.duration_seconds)}{" "}
                            · {r.song.playback_count.toLocaleString()} plays
                          </div>
                        </div>
                        <div className="flex items-center gap-2 text-[10px]">
                          <span className="rounded-full bg-zinc-800 px-2 py-1 uppercase tracking-wide text-zinc-300">
                            {a.energy} energy
                          </span>
                          <span className="rounded-full bg-zinc-800 px-2 py-1 uppercase tracking-wide text-zinc-300">
                            {a.tempo_feel} · {a.bpm_estimate}bpm
                          </span>
                          {a.has_vocals ? (
                            <span className="rounded-full bg-zinc-800 px-2 py-1 uppercase tracking-wide text-zinc-300">
                              {a.vocal_type} vocals
                            </span>
                          ) : (
                            <span className="rounded-full bg-emerald-900/60 px-2 py-1 uppercase tracking-wide text-emerald-300">
                              instrumental
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="mt-2 flex flex-wrap gap-1">
                        {a.moods.slice(0, 5).map((m) => (
                          <span
                            key={m}
                            className="rounded-full bg-indigo-950 px-2 py-0.5 text-[10px] text-indigo-300"
                          >
                            {m}
                          </span>
                        ))}
                        {a.instruments.slice(0, 4).map((i) => (
                          <span
                            key={i}
                            className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300"
                          >
                            {i}
                          </span>
                        ))}
                        {a.genres.slice(0, 3).map((g) => (
                          <span
                            key={g}
                            className="rounded-full bg-amber-950 px-2 py-0.5 text-[10px] text-amber-300"
                          >
                            {g}
                          </span>
                        ))}
                        {(a.themes ?? []).slice(0, 2).map((t) => (
                          <span
                            key={t}
                            className="rounded-full bg-violet-950 px-2 py-0.5 text-[10px] text-violet-300"
                          >
                            {t}
                          </span>
                        ))}
                        {a.sentiment && (
                          <span className="rounded-full bg-rose-950 px-2 py-0.5 text-[10px] text-rose-300">
                            {a.sentiment}
                          </span>
                        )}
                      </div>

                      <audio
                        controls
                        preload="none"
                        src={r.song.audio_url}
                        onPlay={(e) => {
                          document.querySelectorAll("audio").forEach((el) => {
                            if (el !== e.currentTarget) el.pause();
                          });
                        }}
                        className="mt-3 h-9 w-full"
                      />

                      <div className="mt-2 flex items-center justify-between gap-2 text-[11px] text-zinc-500">
                        <span className="truncate">{r.why}</span>
                        <button
                          onClick={() =>
                            setExpanded(isOpen ? null : r.song.song_id)
                          }
                          className="shrink-0 text-emerald-400 hover:text-emerald-300"
                        >
                          {isOpen ? "less" : "details"}
                        </button>
                      </div>

                      {isOpen && (
                        <div className="mt-3 space-y-2 border-t border-zinc-800 pt-3 text-xs text-zinc-400">
                          <p>{a.description}</p>
                          <p className="text-zinc-500">{a.production}</p>
                          {a.lyrics_gist && (
                            <p>
                              <span className="font-semibold text-zinc-300">
                                Lyrics:
                              </span>{" "}
                              {a.lyrics_gist}
                            </p>
                          )}
                          {(a.themes?.length || a.sentiment) && (
                            <p>
                              <span className="font-semibold text-zinc-300">
                                Themes:
                              </span>{" "}
                              {(a.themes ?? []).join(", ")}
                              {a.sentiment && ` · ${a.sentiment}`}
                            </p>
                          )}
                          {a.lyrics_translation_en && (
                            <p className="text-zinc-500">
                              <span className="font-semibold text-zinc-300">
                                Lyrics (EN):
                              </span>{" "}
                              {a.lyrics_translation_en.slice(0, 220)}
                              {a.lyrics_translation_en.length > 220 && "…"}
                            </p>
                          )}
                          {a.sounds_like.length > 0 && (
                            <p>
                              <span className="font-semibold text-zinc-300">
                                Sounds like:
                              </span>{" "}
                              {a.sounds_like.join(", ")}
                            </p>
                          )}
                          {a.use_cases.length > 0 && (
                            <p>
                              <span className="font-semibold text-zinc-300">
                                Good for:
                              </span>{" "}
                              {a.use_cases.join(", ")}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
            )}
          </div>
        )}

        {!data && !error && (
          <div className="mt-16 text-center text-sm text-zinc-500">
            <span className="text-zinc-300">149 tracks · 50 artists · 4 languages</span> — every
            track&apos;s audio analysed by AI: instruments, mood, energy, full lyrics
            (translated), production style.
            <br />
            Search by meaning in any language. Toggle{" "}
            <span className="text-zinc-300">keyword only</span> to see what search
            misses without AI.
          </div>
        )}
      </div>
    </div>
  );
}
