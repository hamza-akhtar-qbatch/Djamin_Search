import { NextResponse } from "next/server";
import { getIndex } from "@/lib/searchEngine";

export const dynamic = "force-dynamic";

export async function GET() {
  const { songs, artists } = getIndex();
  const languages = new Set(
    songs.map((s) => s.analysis.language).filter((l) => l && l !== "none")
  );
  const translated = songs.filter((s) => s.analysis.lyrics_translation_en).length;
  const instrumental = songs.filter((s) => !s.analysis.has_vocals).length;
  return NextResponse.json({
    songs: songs.length,
    artists: artists.length,
    languages: languages.size,
    translated,
    instrumental,
  });
}
