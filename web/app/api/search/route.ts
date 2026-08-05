import { NextRequest, NextResponse } from "next/server";
import { search, reloadIndex } from "@/lib/searchEngine";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const { query, mode, reload } = await req.json();
    if (reload) reloadIndex();
    if (!query || typeof query !== "string" || !query.trim()) {
      return NextResponse.json({ error: "query required" }, { status: 400 });
    }
    const m = mode === "keyword" ? "keyword" : "hybrid";
    const out = await search(query.trim(), m);
    return NextResponse.json(out);
  } catch (e) {
    console.error(e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "search failed" },
      { status: 500 }
    );
  }
}
