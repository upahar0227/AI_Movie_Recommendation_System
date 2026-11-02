import { NextResponse } from "next/server";
import { createRecommendations } from "@/lib/recommend";
import type { RecommendRequest } from "@/lib/types";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Partial<RecommendRequest>;
    const payload: RecommendRequest = {
      mood: typeof body.mood === "string" ? body.mood.trim() : "",
      favorites: typeof body.favorites === "string" ? body.favorites.trim() : "",
      avoid: typeof body.avoid === "string" ? body.avoid.trim() : "",
      era:
        body.era === "classic" || body.era === "2000s" || body.era === "recent"
          ? body.era
          : "any",
    };

    if (!payload.mood && !payload.favorites) {
      return NextResponse.json(
        { error: "Tell me a mood or a few favorite movies first." },
        { status: 400 },
      );
    }

    const result = await createRecommendations(payload);
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Recommendation failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
