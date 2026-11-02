import OpenAI from "openai";
import { recommendFromCatalog } from "./catalog";
import type { MovieRecommendation, RecommendRequest, RecommendResponse } from "./types";

function isRecommendation(value: unknown): value is MovieRecommendation {
  if (!value || typeof value !== "object") return false;
  const movie = value as MovieRecommendation;
  return (
    typeof movie.title === "string" &&
    typeof movie.year === "number" &&
    Array.isArray(movie.genres) &&
    typeof movie.vibe === "string" &&
    typeof movie.why === "string" &&
    typeof movie.match === "number"
  );
}

export async function createRecommendations(
  request: RecommendRequest,
): Promise<RecommendResponse> {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    return { source: "catalog", ...recommendFromCatalog(request) };
  }

  const client = new OpenAI({ apiKey });
  const model = process.env.OPENAI_MODEL || "gpt-4o-mini";

  const completion = await client.chat.completions.create({
    model,
    temperature: 0.8,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You are an expert film programmer. Recommend 6 real movies.
Return JSON with keys summary (string) and recommendations (array).
Each recommendation must include title, year, genres (array of 1-3 strings), vibe (short phrase), why (1-2 sentences, no spoilers), and match (integer 60-99).
Do not invent fake films. Prefer a mix of well-known and slightly underseen titles.`,
      },
      {
        role: "user",
        content: `Mood or night: ${request.mood || "open to anything"}
Favorite movies: ${request.favorites || "none given"}
Avoid: ${request.avoid || "nothing specific"}
Era preference: ${request.era}`,
      },
    ],
  });

  const raw = completion.choices[0]?.message?.content;
  if (!raw) {
    return { source: "catalog", ...recommendFromCatalog(request) };
  }

  const parsed = JSON.parse(raw) as {
    summary?: string;
    recommendations?: unknown[];
  };
  const recommendations = (parsed.recommendations ?? []).filter(isRecommendation).slice(0, 6);

  if (recommendations.length === 0) {
    return { source: "catalog", ...recommendFromCatalog(request) };
  }

  return {
    source: "openai",
    summary: parsed.summary || "A personal slate based on what you described.",
    recommendations,
  };
}
