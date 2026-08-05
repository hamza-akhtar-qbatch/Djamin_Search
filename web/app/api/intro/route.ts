import { NextRequest, NextResponse } from "next/server";
import { draftIntro } from "@/lib/searchEngine";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const { artist_id, query } = await req.json();
    if (!artist_id) {
      return NextResponse.json({ error: "artist_id required" }, { status: 400 });
    }
    const message = await draftIntro(artist_id, query ?? "");
    return NextResponse.json({ message });
  } catch (e) {
    console.error(e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "intro failed" },
      { status: 500 }
    );
  }
}
