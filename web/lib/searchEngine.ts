/**
 * Djaminn hybrid search engine.
 *
 * Pipeline: query -> Gemini query understanding (intent + filters + semantic
 * rewrite) -> parallel retrieval (vector cosine over audio-profile embeddings
 * + BM25 over text fields) -> reciprocal-rank fusion -> metadata filter
 * boosts -> engagement/recency prior blend -> per-artist diversity ->
 * explainable results. `mode: "keyword"` skips the AI half so the demo can
 * show the difference live.
 */

import fs from "fs";
import path from "path";

const GEN_MODEL = "gemini-3.1-flash-lite";
const EMBED_MODEL = "gemini-embedding-001";
const API = "https://generativelanguage.googleapis.com/v1beta/models";

// ---------------------------------------------------------------- types

export interface SongAnalysis {
  description: string;
  transcription: string;
  lyrics_gist: string;
  instruments: string[];
  moods: string[];
  genres: string[];
  energy: "low" | "medium" | "high";
  has_vocals: boolean;
  vocal_type: string;
  vocal_delivery: string;
  language: string;
  bpm_estimate: number;
  tempo_feel: "slow" | "mid" | "fast";
  key_guess: string;
  production: string;
  is_cover_likely: boolean;
  good_for_remixing: boolean;
  use_cases: string[];
  sounds_like: string[];
  // second-stage multilingual pass (optional on older records)
  lyrics_translation_en?: string;
  themes?: string[];
  sentiment?: string;
}

export interface Song {
  song_id: string;
  title: string;
  artist: string;
  artist_id: string;
  audio_url: string;
  upload_date: string;
  playback_count: number;
  duration_seconds: number;
  region: string;
  artist_genres: string[];
  analysis: SongAnalysis;
  profile_text: string;
  embedding: number[];
  prior: number;
}

export interface QueryPlan {
  semantic_query: string;
  keywords: string[];
  filters: {
    has_vocals?: boolean;
    vocal_type?: string;
    energy?: string;
    tempo_feel?: string;
    language?: string;
    good_for_remixing?: boolean;
  };
  result_focus: "songs" | "artists" | "both";
  intent: "search" | "popular_songs" | "new_artists" | "collaborators" | "near_me" | "remix_material";
  location_mention?: string;
  is_music_related?: boolean;
  unanswerable_aspects?: string[];
}

export interface Artist {
  artist_id: string;
  name: string;
  region: string | null;
  rating: number;
  rank: number;
  follow_count: number;
  follower_count: number;
  total_playback: number;
  song_count: number;
  genres: string[];
  enriched_songs: number;
  first_seen: string | null;
  last_seen: string | null;
  export_capped: boolean;
  top_moods: string[];
  top_instruments: string[];
  is_house: boolean;
}

export interface SearchResult {
  song: Omit<Song, "embedding" | "profile_text">;
  score: number;
  semanticScore: number | null;
  keywordTerms: string[];
  matchedFilters: string[];
  why: string;
}

export interface ArtistResult {
  artist: string;
  artist_id: string;
  score: number;
  songCount: number;
  genres: string[];
  topMoods: string[];
  topInstruments: string[];
  evidence: string[];
}

// ---------------------------------------------------------------- index

interface Index {
  songs: Song[];
  artists: Artist[];
  anchorDate: Date;
  bm25: Bm25Index;
  folded: string[];
}

let _index: Index | null = null;
let _indexMtime = 0;

export function getIndex(): Index {
  const file = path.join(process.cwd(), "data", "index.json");
  // auto-reload when the compiled index changes on disk — no flag needed,
  // and no wasteful BM25 rebuild when it hasn't
  const mtime = fs.statSync(file).mtimeMs;
  if (_index && mtime === _indexMtime) return _index;
  _indexMtime = mtime;
  const raw = JSON.parse(fs.readFileSync(file, "utf-8"));
  const songs: Song[] = raw.songs;
  _index = {
    songs,
    artists: raw.artists ?? [],
    anchorDate: new Date(raw.anchor_date),
    bm25: buildBm25(songs),
    // folded title+lyrics per song, so a pasted lyric line can match as an
    // exact phrase and outrank token-coincidence results
    folded: songs.map((s) =>
      fold(`${s.title} | ${s.analysis.transcription} | ${s.analysis.lyrics_translation_en ?? ""}`)
    ),
  };
  return _index;
}

/** Force reload (after re-running enrichment). */
export function reloadIndex() {
  _index = null;
}

// ---------------------------------------------------------------- text utils

function fold(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "");
}

const STOP = new Set([
  "to", "the", "a", "an", "of", "for", "in", "on", "at", "and", "or", "is",
  "with", "by", "me", "my", "you", "it", "de", "da", "do", "em", "um", "uma",
  "el", "la", "los", "las", "que", "y", "e", "o", "no", "na",
  // query meta-words: describe what the user wants, not the content
  "about", "song", "songs", "music", "sounds", "sound", "find", "want",
  "need", "some", "something", "very", "really",
]);

function tokenize(s: string): string[] {
  return fold(s)
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 1 && !STOP.has(t));
}

// ---------------------------------------------------------------- BM25

interface Bm25Index {
  docTokens: Map<number, Map<string, number>>;
  docLen: number[];
  avgLen: number;
  df: Map<string, number>;
  N: number;
}

function buildBm25(songs: Song[]): Bm25Index {
  const docTokens = new Map<number, Map<string, number>>();
  const docLen: number[] = [];
  const df = new Map<string, number>();
  songs.forEach((s, i) => {
    // title + artist weighted higher by repetition; analysis text provides
    // the AI-derived vocabulary (moods, instruments, lyrics)
    const a = s.analysis;
    const text = [
      s.title, s.title, s.title,
      s.artist, s.artist,
      a.description, a.lyrics_gist, a.transcription.slice(0, 1200),
      a.lyrics_translation_en?.slice(0, 1200) ?? "",
      (a.themes ?? []).join(" "), a.sentiment ?? "",
      a.instruments.join(" "), a.moods.join(" "), a.genres.join(" "),
      a.use_cases.join(" "), a.sounds_like.join(" "),
      s.artist_genres.join(" "),
    ].join(" ");
    const counts = new Map<string, number>();
    const toks = tokenize(text);
    for (const t of toks) counts.set(t, (counts.get(t) ?? 0) + 1);
    docTokens.set(i, counts);
    docLen.push(toks.length);
    for (const t of counts.keys()) df.set(t, (df.get(t) ?? 0) + 1);
  });
  const avgLen = docLen.reduce((x, y) => x + y, 0) / docLen.length;
  return { docTokens, docLen, avgLen, df, N: songs.length };
}

/** Bounded Levenshtein distance with early exit above `max`. */
function editDistance(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(
        prev[j] + 1,
        cur[j - 1] + 1,
        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
      rowMin = Math.min(rowMin, cur[j]);
    }
    if (rowMin > max) return max + 1;
    prev = cur;
  }
  return prev[b.length];
}

/** Typo tolerance: query terms with zero index hits get their closest
 *  vocabulary neighbour appended ("gitar" -> "guitar"). Max 3 corrections. */
function fuzzyExpand(idx: Bm25Index, terms: string[]): string[] {
  const out = [...terms];
  let corrections = 0;
  for (const t of terms) {
    if (corrections >= 3 || t.length < 4 || idx.df.has(t)) continue;
    const maxDist = t.length >= 7 ? 2 : 1;
    let best: string | null = null;
    let bestDf = 0;
    for (const [vocab, df] of idx.df) {
      if (Math.abs(vocab.length - t.length) > maxDist) continue;
      if (editDistance(t, vocab, maxDist) <= maxDist && df > bestDf) {
        best = vocab;
        bestDf = df;
      }
    }
    if (best) {
      out.push(best);
      corrections++;
    }
  }
  return out;
}

function bm25Scores(idx: Bm25Index, terms: string[]): Map<number, { score: number; matched: Set<string> }> {
  const k1 = 1.4, b = 0.75;
  const out = new Map<number, { score: number; matched: Set<string> }>();
  for (const term of new Set(terms)) {
    const n = idx.df.get(term);
    if (!n) continue;
    const idf = Math.log(1 + (idx.N - n + 0.5) / (n + 0.5));
    idx.docTokens.forEach((counts, docId) => {
      const tf = counts.get(term);
      if (!tf) return;
      const denom = tf + k1 * (1 - b + (b * idx.docLen[docId]) / idx.avgLen);
      const s = (idf * (tf * (k1 + 1))) / denom;
      const cur = out.get(docId) ?? { score: 0, matched: new Set<string>() };
      cur.score += s;
      cur.matched.add(term);
      out.set(docId, cur);
    });
  }
  return out;
}

// ---------------------------------------------------------------- gemini

const API_KEYS = [
  process.env.GEMINI_API_KEY,
  process.env.GEMINI_API_KEY_2,
  process.env.GEMINI_API_KEY_3,
  process.env.GEMINI_API_KEY_4,
].filter((k): k is string => !!k);

function apiKey(): string {
  if (!API_KEYS.length) throw new Error("GEMINI_API_KEY not set");
  return API_KEYS[0];
}

async function geminiJson(url: string, body: unknown): Promise<any> {
  for (let attempt = 0; attempt < 5; attempt++) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.ok) return res.json();
    if (res.status === 429) {
      // quota on this key -> roll to the fallback key before backing off
      const cur = url.match(/key=([^&]+)/)?.[1];
      const idx = cur ? API_KEYS.indexOf(cur) : -1;
      if (idx >= 0 && idx + 1 < API_KEYS.length) {
        url = url.replace(`key=${cur}`, `key=${API_KEYS[idx + 1]}`);
        continue;
      }
    }
    if ([429, 500, 502, 503].includes(res.status) && attempt < 4) {
      await new Promise((r) => setTimeout(r, 2 ** attempt * 1500));
      continue;
    }
    throw new Error(`Gemini ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  throw new Error("Gemini retries exhausted");
}

const PLAN_SCHEMA = {
  type: "object",
  properties: {
    semantic_query: {
      type: "string",
      description:
        "Rewrite of the user's query as a rich description of the music they want, suitable for embedding similarity against audio descriptions. Expand mood/feel/instrument implications.",
    },
    keywords: {
      type: "array",
      items: { type: "string" },
      description: "Exact-match terms worth keyword-searching: names, titles, instruments, genres, distinctive words. Lowercase.",
    },
    filters: {
      type: "object",
      properties: {
        has_vocals: { type: "boolean" },
        vocal_type: { type: "string", enum: ["male", "female", "group"] },
        energy: { type: "string", enum: ["low", "medium", "high"] },
        tempo_feel: { type: "string", enum: ["slow", "mid", "fast"] },
        language: { type: "string", description: "ISO 639-1" },
        good_for_remixing: { type: "boolean" },
      },
    },
    result_focus: {
      type: "string",
      enum: ["songs", "artists", "both"],
      description: "'artists' when the user seeks people/roles (drummer, collaborators), 'songs' for music, 'both' when ambiguous",
    },
    intent: {
      type: "string",
      enum: ["search", "popular_songs", "new_artists", "collaborators", "near_me", "remix_material"],
      description:
        "'popular_songs' = most played/viewed/liked music; 'new_artists' = recently joined artists; " +
        "'collaborators' = ONLY a fully generic ask for people to work with ('collaborators', 'someone to jam with') " +
        "with NO role, genre, gender or style mentioned. The moment the user names WHAT they need — " +
        "'drummer to collaborate with', 'female singer for my bossa track', 'guitarist for a rock project' — " +
        "use 'search' with result_focus='artists' so content matching finds the right people; " +
        "'near_me' = location-based artist search; 'remix_material' = instrumentals/beats/loops to remix or build on; " +
        "otherwise 'search'. Role/instrument words alone (drummer, vocalist, pianist) are also 'search' with result_focus='artists'.",
    },
    location_mention: {
      type: "string",
      description: "Place the user named (e.g. 'latin america', 'asia'), if any. Omit for 'near me' with no explicit place.",
    },
    is_music_related: {
      type: "boolean",
      description:
        "false ONLY if the input is clearly not a music search at all (random characters, field names like 'search_text', test strings, gibberish). Any plausible musical meaning, in any language -> true.",
    },
    unanswerable_aspects: {
      type: "array",
      items: { type: "string" },
      description:
        "Aspects of the query the catalog CANNOT answer, as short phrases. The catalog KNOWS: audio-derived instruments/moods/genres/energy/tempo, singing voice type incl. male/female (so 'female singer'/'female vocals' IS answerable — never list it), lyrics + translations, themes, sentiment, coarse region, play counts, activity. It does NOT know: gender of instrumentalists (drummer/guitarist gender), age, exact city, skill level. ONLY list aspects the user explicitly asked about — never list something the query doesn't mention. e.g. 'female drummer' -> ['drummer gender']; 'female singer' -> []; 'drummer' -> []. Empty when everything is answerable.",
    },
  },
  required: ["semantic_query", "keywords", "filters", "result_focus", "intent", "is_music_related"],
};

const planCache = new Map<string, QueryPlan>();

/** Draft a short collaboration intro from the searcher to a found artist,
 *  grounded in the artist's actual tracks — the "connect" step after search. */
export async function draftIntro(artistId: string, query: string): Promise<string> {
  const { songs, artists } = getIndex();
  const artist = artists.find((a) => a.artist_id === artistId);
  const tracks = songs
    .filter((s) => s.artist_id === artistId)
    .sort((a, b) => b.playback_count - a.playback_count)
    .slice(0, 3);
  const name = artist?.name ?? tracks[0]?.artist ?? "there";
  const trackLines = tracks
    .map((s) => {
      const a = s.analysis;
      return `- "${s.title}": ${a.description.slice(0, 160)} (moods: ${a.moods.slice(0, 3).join(", ")})`;
    })
    .join("\n");
  const r = await geminiJson(`${API}/${GEN_MODEL}:generateContent?key=${apiKey()}`, {
    contents: [
      {
        parts: [
          {
            text:
              `A musician on the Djaminn platform searched for: "${query}" and found the artist ${name}` +
              `${artist?.region ? ` (${artist.region})` : ""}.\n` +
              `${name}'s tracks:\n${trackLines}\n\n` +
              `Write the first message the searcher could send ${name} to start a collaboration. ` +
              `2-3 sentences. Mention one of their actual tracks and what fits. ` +
              `Warm but plain, like a real musician typing on their phone. No emojis, no exclamation marks, ` +
              `start with "Hi ${name}". Return only the message text.`,
          },
        ],
      },
    ],
    generationConfig: { temperature: 0.4 },
  });
  return r.candidates[0].content.parts[0].text.trim();
}

export async function understandQuery(query: string): Promise<QueryPlan> {
  const key = fold(query.trim());
  const cached = planCache.get(key);
  if (cached) return cached;
  try {
    const r = await geminiJson(`${API}/${GEN_MODEL}:generateContent?key=${apiKey()}`, {
      contents: [
        {
          parts: [
            {
              text:
                `A musician typed this into a music collaboration platform's search box: "${query}"\n` +
                `Interpret what they actually want. The catalog is user-uploaded songs (many in Portuguese/Spanish/English/Thai) ` +
                `with AI audio analysis: instruments, moods, energy, tempo, vocals, genres, lyrics. ` +
                `Set a filter ONLY when the user explicitly asked for that attribute (e.g. "female vocals" -> has_vocals=true, vocal_type=female; ` +
                `"instrumental" -> has_vocals=false; "chill"/"slow" -> energy/tempo). Do NOT infer tempo or energy from genre or vibe words ` +
                `like "party", "hit", "reggaeton" — genres carry their own tempo conventions the catalog knows better than you.`,
            },
          ],
        },
      ],
      generationConfig: {
        responseMimeType: "application/json",
        responseJsonSchema: PLAN_SCHEMA,
        temperature: 0,
      },
    });
    const plan: QueryPlan = JSON.parse(r.candidates[0].content.parts[0].text);
    planCache.set(key, plan);
    return plan;
  } catch {
    // graceful fallback: plain hybrid search without filters
    return { semantic_query: query, keywords: tokenize(query), filters: {}, result_focus: "both", intent: "search" };
  }
}

const embedCache = new Map<string, number[]>();

export async function embedQuery(text: string): Promise<number[]> {
  const cached = embedCache.get(text);
  if (cached) return cached;
  const r = await geminiJson(`${API}/${EMBED_MODEL}:embedContent?key=${apiKey()}`, {
    content: { parts: [{ text }] },
    taskType: "RETRIEVAL_QUERY",
    outputDimensionality: 768,
  });
  const v: number[] = r.embedding.values;
  embedCache.set(text, v);
  return v;
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ---------------------------------------------------------------- search

function filterMatch(song: Song, f: QueryPlan["filters"]): { boost: number; matched: string[] } {
  let boost = 1;
  const matched: string[] = [];
  const a = song.analysis;
  const checks: [string, boolean | undefined, boolean][] = [
    ["vocals", f.has_vocals, f.has_vocals !== undefined && a.has_vocals === f.has_vocals],
    ["vocal type", f.vocal_type !== undefined ? true : undefined, a.vocal_type === f.vocal_type],
    ["energy", f.energy !== undefined ? true : undefined, a.energy === f.energy],
    ["tempo", f.tempo_feel !== undefined ? true : undefined, a.tempo_feel === f.tempo_feel],
    ["language", f.language !== undefined ? true : undefined, a.language === f.language],
    ["remixable", f.good_for_remixing, f.good_for_remixing !== undefined && a.good_for_remixing === f.good_for_remixing],
  ];
  for (const [label, wanted, ok] of checks) {
    if (wanted === undefined) continue;
    if (ok) {
      boost *= 1.2;
      matched.push(label);
    } else {
      // soft nudge, never a veto: filters are planner inferences, and a
      // strong content match (e.g. the only reggaeton track) must survive
      // a wrong tempo/energy guess
      boost *= 0.8;
    }
  }
  return { boost, matched };
}

interface SearchResponse {
  plan: QueryPlan;
  results: SearchResult[];
  artists: ArtistResult[];
  note?: string;
  timings: Record<string, number>;
}

function songToResult(s: Song, why: string): SearchResult {
  const { embedding: _e, profile_text: _p, ...songOut } = s;
  return { song: songOut, score: 0, semanticScore: null, keywordTerms: [], matchedFilters: [], why };
}

function artistCard(a: Artist, evidence: string[], score = 0): ArtistResult {
  return {
    artist: a.name,
    artist_id: a.artist_id,
    score,
    songCount: a.enriched_songs || a.song_count,
    genres: a.genres.slice(0, 5),
    topMoods: a.top_moods.slice(0, 4),
    topInstruments: a.top_instruments.slice(0, 4),
    evidence,
  };
}

const daysBetween = (a: Date, b: Date) => Math.max(0, (a.getTime() - b.getTime()) / 86400000);

/** Intents that rank on structured signals (plays, activity, social counts,
 *  region, audio-derived remixability) instead of text similarity. */
function structuredIntent(
  plan: QueryPlan,
  _query: string
): { results: SearchResult[]; artists: ArtistResult[]; note: string } | null {
  const { songs, artists, anchorDate } = getIndex();

  const songsByArtist = new Map<string, Song[]>();
  for (const s of songs) {
    const l = songsByArtist.get(s.artist_id) ?? [];
    l.push(s);
    songsByArtist.set(s.artist_id, l);
  }
  const artistById = new Map(artists.map((a) => [a.artist_id, a]));
  /** Artists behind a set of song results — so every search that matches
   *  songs also shows who those songs belong to. */
  const relatedArtists = (rs: SearchResult[], max = 5): ArtistResult[] => {
    const seen = new Set<string>();
    const out: ArtistResult[] = [];
    for (const r of rs) {
      const id = r.song.artist_id;
      if (seen.has(id)) continue;
      seen.add(id);
      const a = artistById.get(id);
      if (!a || a.is_house) continue;
      out.push(artistCard(a, [`matches: ${r.song.title}`]));
      if (out.length >= max) break;
    }
    return out;
  };
  /** Songs section content for artist-centric intents: each listed artist's
   *  best (or newest) tracks, so every search also surfaces playable music. */
  const topSongsFor = (
    ids: string[],
    why: (s: Song) => string,
    sortBy: "plays" | "recent" = "plays",
    per = 1,
    cap = 9
  ): SearchResult[] => {
    const out: SearchResult[] = [];
    for (const id of ids) {
      const list = (songsByArtist.get(id) ?? [])
        .slice()
        .sort((a, b) =>
          sortBy === "plays"
            ? b.playback_count - a.playback_count
            : b.upload_date.localeCompare(a.upload_date)
        )
        .slice(0, per);
      for (const s of list) {
        out.push(songToResult(s, why(s)));
        if (out.length >= cap) return out;
      }
    }
    return out;
  };

  if (plan.intent === "popular_songs") {
    const ranked = [...songs].sort((a, b) => b.playback_count - a.playback_count);
    const per = new Map<string, number>();
    const out: SearchResult[] = [];
    for (const s of ranked) {
      const n = per.get(s.artist_id) ?? 0;
      if (n >= 2) continue;
      per.set(s.artist_id, n + 1);
      out.push(songToResult(s, `${s.playback_count.toLocaleString()} plays · uploaded ${s.upload_date.slice(0, 10)}`));
      if (out.length >= 12) break;
    }
    return {
      results: out,
      artists: relatedArtists(out),
      note: "Ranked by lifetime playback, max 2 per artist. Lifetime counts favour old uploads — production should rank on windowed play events (e.g. last 30 days), which the export doesn't include.",
    };
  }

  if (plan.intent === "new_artists") {
    const ranked = artists
      .filter((a) => a.first_seen && !a.is_house)
      .sort((a, b) => b.first_seen!.localeCompare(a.first_seen!));
    const top = ranked.slice(0, 9);
    return {
      results: topSongsFor(top.map((a) => a.artist_id), (s) => `latest track by ${s.artist} · uploaded ${s.upload_date.slice(0, 10)}`, "recent"),
      artists: top.map((a) =>
        artistCard(a, [
          `first upload ${a.first_seen!.slice(0, 10)}`,
          ...(a.export_capped ? ["export capped — true join date unknown"] : []),
        ])
      ),
      note: "No signup date exists in the data — 'new' is proxied by earliest upload in the export. The platform should expose account creation dates to answer this properly.",
    };
  }

  if (plan.intent === "collaborators") {
    const scored = artists
      .filter((a) => !a.is_house && a.last_seen)
      .map((a) => {
        const idle = daysBetween(anchorDate, new Date(a.last_seen!));
        const active = Math.exp(-idle / 45);
        const social = Math.min(Math.log1p(a.follow_count) / Math.log(101), 1);
        const wanted = Math.min(Math.log1p(a.follower_count) / Math.log(3001), 1);
        return {
          a,
          score: 0.5 * active + 0.3 * social + 0.2 * wanted,
          why: [`active ${Math.round(idle)}d before export`, `follows ${a.follow_count}`, `${a.follower_count} followers`],
        };
      })
      .sort((x, y) => y.score - x.score)
      .slice(0, 9);
    return {
      results: topSongsFor(scored.map((s) => s.a.artist_id), (s) => `top track by ${s.artist} · ${s.playback_count.toLocaleString()} plays`),
      artists: scored.map((s) => artistCard(s.a, s.why, s.score)),
      note: "Proxy ranking: recently active + follows others + is followed. True collaboration matching needs project/invite/response data — and complementary-role matching once instrument roles exist.",
    };
  }

  if (plan.intent === "near_me") {
    const loc = plan.location_mention ? tokenize(plan.location_mention) : [];
    if (loc.length) {
      const hits = artists
        .filter((a) => a.region && !a.is_house && loc.some((t) => fold(a.region!).includes(t)))
        .sort((a, b) => b.rating - a.rating);
      if (hits.length) {
        const top = hits.slice(0, 9);
        return {
          results: topSongsFor(top.map((a) => a.artist_id), (s) => `by ${s.artist} · ${s.region ?? ""}`),
          artists: top.map((a) => artistCard(a, [a.region!, `rating ${a.rating}`])),
          note: `Matched region bucket for "${plan.location_mention}". Regions are 6 coarse buckets — true proximity needs opt-in city-level location on profiles.`,
        };
      }
    }
    const counts = new Map<string, number>();
    for (const a of artists) if (a.region) counts.set(a.region, (counts.get(a.region) ?? 0) + 1);
    const breakdown = [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([r, n]) => `${r} (${n})`)
      .join(" · ");
    // no usable location: still surface music — the top artist per region
    const regionReps: string[] = [];
    for (const [region] of [...counts.entries()].sort((a, b) => b[1] - a[1])) {
      const rep = artists
        .filter((a) => a.region === region && !a.is_house && (songsByArtist.get(a.artist_id)?.length ?? 0) > 0)
        .sort((a, b) => b.rating - a.rating)[0];
      if (rep) regionReps.push(rep.artist_id);
    }
    return {
      results: topSongsFor(regionReps, (s) => `top artist in ${s.region ?? "region"}`),
      artists: [],
      note: `"Near me" needs your location, and profiles only carry 6 coarse region buckets: ${breakdown}. Showing each region's top artist — try "artists in latin america". Real proximity search needs opt-in city/geo capture.`,
    };
  }

  if (plan.intent === "remix_material") {
    const ranked = songs
      .filter((s) => s.analysis.good_for_remixing)
      .map((s) => ({ s, score: (s.analysis.has_vocals ? 0.6 : 1) * (1 + s.prior) }))
      .sort((a, b) => b.score - a.score);
    const per = new Map<string, number>();
    const out: SearchResult[] = [];
    for (const { s } of ranked) {
      const n = per.get(s.artist_id) ?? 0;
      if (n >= 2) continue;
      per.set(s.artist_id, n + 1);
      out.push(
        songToResult(
          s,
          `AI-flagged remixable · ${s.analysis.has_vocals ? "has vocals" : "instrumental"} · ${s.analysis.moods.slice(0, 3).join(", ")}`
        )
      );
      if (out.length >= 12) break;
    }
    return {
      results: out,
      artists: relatedArtists(out),
      note: "Ranked from the audio analysis itself: tracks the AI judged to be loop/beat/instrumental material a producer could build on. Vocal tracks are downranked (rights + usability).",
    };
  }

  return null;
}

export async function search(
  query: string,
  mode: "hybrid" | "keyword" = "hybrid",
  limit = 12
): Promise<SearchResponse> {
  const t0 = Date.now();
  const { songs, bm25 } = getIndex();

  const plan: QueryPlan =
    mode === "hybrid"
      ? await understandQuery(query)
      : { semantic_query: query, keywords: tokenize(query), filters: {}, result_focus: "both", intent: "search" };
  const t1 = Date.now();

  // a query with no musical meaning must return empty, not the nearest noise
  if (plan.is_music_related === false) {
    return {
      plan,
      results: [],
      artists: [],
      note: "This doesn't look like a music search — nothing in the catalog relates to it.",
      timings: { planMs: t1 - t0, retrievalMs: 0, totalMs: Date.now() - t0 },
    };
  }

  // aspects the data cannot answer are said out loud, not silently dropped.
  // Guard against planner over-flagging: a gender aspect only counts when the
  // query actually mentions gender.
  const qLower = fold(query);
  const mentionsGender = /\b(female|male|woman|women|man|men|girl|boy|guy|lady)\b/.test(qLower);
  const aspects = (plan.unanswerable_aspects ?? []).filter(
    (a) => !/gender|sex/i.test(a) || mentionsGender
  );
  const aspectsNote = aspects.length
    ? `Not in the data (yet): ${aspects.join(", ")} — results ignore that part of the query.`
    : undefined;

  // structured intents rank on real signals (plays, activity, social counts,
  // region, audio-derived remixability) instead of text similarity
  if (plan.intent !== "search") {
    const structured = structuredIntent(plan, query);
    if (structured) {
      const note = [aspectsNote, structured.note].filter(Boolean).join(" ");
      return { ...structured, note, plan, timings: { planMs: t1 - t0, retrievalMs: 0, totalMs: Date.now() - t0 } };
    }
  }

  // keyword retrieval (always on — exact names/titles must never lose)
  const kwTerms = fuzzyExpand(bm25, [
    ...new Set([...tokenize(query), ...plan.keywords.flatMap(tokenize)]),
  ]);
  const kw = bm25Scores(bm25, kwTerms);
  const kwRanked = [...kw.entries()].sort((a, b) => b[1].score - a[1].score);

  // semantic retrieval — degrade to keyword-only rather than fail the search
  let semRanked: [number, number][] = [];
  let degradedNote: string | undefined;
  if (mode === "hybrid") {
    try {
      const qVec = await embedQuery(plan.semantic_query);
      semRanked = songs
        .map((s, i) => [i, cosine(qVec, s.embedding)] as [number, number])
        .sort((a, b) => b[1] - a[1]);
    } catch {
      degradedNote = "Semantic search is momentarily unavailable — showing keyword results.";
    }
  }
  const t2 = Date.now();

  // reciprocal-rank fusion
  const K = 60;
  const fused = new Map<number, number>();
  kwRanked.forEach(([docId], rank) => {
    if (rank < 50) fused.set(docId, (fused.get(docId) ?? 0) + 1 / (K + rank + 1));
  });
  // relevance floor: below this a song is not actually about the query —
  // an unrelated query must return nothing, not noise. Query-relative: a
  // strong top match raises the bar for everything below it.
  const topSim = semRanked.length ? semRanked[0][1] : 0;
  const SEM_FLOOR = Math.max(0.45, topSim * 0.82);
  semRanked.forEach(([docId, sim], rank) => {
    if (rank < 50 && sim >= SEM_FLOOR) {
      // rank-based RRF plus an absolute-similarity bonus so a genuinely
      // strong meaning match beats a coincidental keyword hit
      const bonus = Math.max(0, sim - 0.5) * 0.03;
      fused.set(docId, (fused.get(docId) ?? 0) + 1.15 / (K + rank + 1) + bonus);
    }
  });

  const semByDoc = new Map(semRanked.map(([d, s]) => [d, s]));
  const { folded } = getIndex();
  const qFold = fold(query.trim());
  const qTermCount = kwTerms.length;
  const scored = [...fused.entries()].map(([docId, base]) => {
    const song = songs[docId];
    const { boost, matched } = filterMatch(song, plan.filters);
    // a pasted lyric line or exact title must beat token coincidences
    const phrase = qFold.length >= 8 && folded[docId].includes(qFold);
    // coverage: matching 4 of 4 distinct query terms must beat 2 of 4 —
    // BM25 length-normalization + RRF alone don't guarantee that
    const kwMatch = kw.get(docId);
    const cov = kwMatch && qTermCount > 1
      ? 1 + 0.35 * ((kwMatch.matched.size - 1) / (qTermCount - 1))
      : 1;
    const score = base * boost * cov * (phrase ? 1.6 : 1) * (1 + 0.35 * song.prior);
    return { docId, song, score, matched, phrase };
  });
  scored.sort((a, b) => b.score - a.score);

  // per-artist diversity cap on songs
  const perArtist = new Map<string, number>();
  const results: SearchResult[] = [];
  for (const r of scored) {
    const n = perArtist.get(r.song.artist_id) ?? 0;
    if (n >= 3) continue;
    perArtist.set(r.song.artist_id, n + 1);
    const sem = semByDoc.get(r.docId) ?? null;
    const kwMatch = kw.get(r.docId);
    const terms = kwMatch ? [...kwMatch.matched].slice(0, 6) : [];
    const whyParts: string[] = [];
    if (r.phrase) whyParts.push("exact phrase in title/lyrics");
    if (sem !== null && sem > 0.3) whyParts.push(`audio meaning match ${(sem * 100).toFixed(0)}%`);
    if (terms.length) whyParts.push(`keywords: ${terms.join(", ")}`);
    if (r.matched.length) whyParts.push(`fits: ${r.matched.join(", ")}`);
    const { embedding: _e, profile_text: _p, ...songOut } = r.song;
    results.push({
      song: songOut,
      score: r.score,
      semanticScore: sem,
      keywordTerms: terms,
      matchedFilters: r.matched,
      why: whyParts.join(" · ") || "engagement + recency",
    });
    if (results.length >= limit) break;
  }

  // artist aggregation from song evidence (house/aggregate accounts excluded —
  // they match everything and aren't a person you can collaborate with)
  const houseIds = new Set(getIndex().artists.filter((a) => a.is_house).map((a) => a.artist_id));
  const byArtist = new Map<string, SearchResult[]>();
  for (const r of scored.slice(0, 40).map((x) => x)) {
    if (houseIds.has(r.song.artist_id)) continue;
    const list = byArtist.get(x_id(r.song)) ?? [];
    const sem = semByDoc.get(r.docId) ?? null;
    const kwMatch = kw.get(r.docId);
    const { embedding: _e2, profile_text: _p2, ...songOut } = r.song;
    list.push({
      song: songOut,
      score: r.score,
      semanticScore: sem,
      keywordTerms: kwMatch ? [...kwMatch.matched] : [],
      matchedFilters: r.matched,
      why: "",
    });
    byArtist.set(x_id(r.song), list);
  }
  const artists: ArtistResult[] = [...byArtist.entries()]
    .map(([aid, rs]) => {
      // mean of top-3 matching songs beats "one lucky song": a consistent
      // specialist outranks a coincidental single match
      const top3 = rs.map((r) => r.score).sort((x, y) => y - x).slice(0, 3);
      const mean3 = top3.reduce((x, y) => x + y, 0) / top3.length;
      const moods = countTop(rs.flatMap((r) => r.song.analysis.moods), 4);
      const instruments = countTop(rs.flatMap((r) => r.song.analysis.instruments), 4);
      return {
        artist: rs[0].song.artist,
        artist_id: aid,
        score: mean3 + 0.01 * Math.min(rs.length, 5),
        songCount: rs.length,
        genres: rs[0].song.artist_genres.slice(0, 5),
        topMoods: moods,
        topInstruments: instruments,
        evidence: rs.slice(0, 3).map((r) => r.song.title),
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  return {
    plan,
    results,
    artists,
    note: [aspectsNote, degradedNote].filter(Boolean).join(" ") || undefined,
    timings: { planMs: t1 - t0, retrievalMs: t2 - t1, totalMs: Date.now() - t0 },
  };
}

function x_id(s: { artist_id: string }): string {
  return s.artist_id;
}

function countTop(items: string[], n: number): string[] {
  const c = new Map<string, number>();
  for (const i of items) c.set(i, (c.get(i) ?? 0) + 1);
  return [...c.entries()].sort((a, b) => b[1] - a[1]).slice(0, n).map(([k]) => k);
}
