"use client";

import { FormEvent, useState } from "react";
import { MovieCard } from "@/components/MovieCard";
import type { RecommendResponse } from "@/lib/types";

const ERAS = [
  { value: "any", label: "Any era" },
  { value: "classic", label: "Before 2000" },
  { value: "2000s", label: "2000–2014" },
  { value: "recent", label: "2015–now" },
] as const;

export function Recommender() {
  const [mood, setMood] = useState("A rainy night, something thoughtful");
  const [favorites, setFavorites] = useState("Arrival, Her, Before Sunset");
  const [avoid, setAvoid] = useState("");
  const [era, setEra] = useState<(typeof ERAS)[number]["value"]>("any");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<RecommendResponse | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mood, favorites, avoid, era }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Could not get recommendations.");
      }
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-6 py-10 lg:px-10">
      <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-xs tracking-[0.28em] text-[#f0b29d]">REELMIND</p>
          <h1 className="mt-3 max-w-2xl font-serif text-5xl leading-[1.05] text-[#fff8ef] md:text-6xl">
            Tell the night what you want to feel.
          </h1>
        </div>
        <p className="max-w-sm text-sm leading-6 text-white/60">
          An AI programmer builds a six-film slate from your mood and favorites. Add an OpenAI key
          for live model picks, or use the built-in catalog right away.
        </p>
      </header>

      <form
        onSubmit={onSubmit}
        className="grid gap-4 rounded-[28px] border border-white/10 bg-black/30 p-5 backdrop-blur-md md:grid-cols-2 lg:grid-cols-4"
      >
        <label className="flex flex-col gap-2 text-sm text-white/70">
          Mood
          <textarea
            value={mood}
            onChange={(event) => setMood(event.target.value)}
            rows={3}
            className="min-h-[88px] resize-none rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-[#fff8ef] outline-none focus:border-[#c45c3e]"
          />
        </label>
        <label className="flex flex-col gap-2 text-sm text-white/70">
          Favorites
          <textarea
            value={favorites}
            onChange={(event) => setFavorites(event.target.value)}
            rows={3}
            className="min-h-[88px] resize-none rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-[#fff8ef] outline-none focus:border-[#c45c3e]"
          />
        </label>
        <label className="flex flex-col gap-2 text-sm text-white/70">
          Avoid
          <textarea
            value={avoid}
            onChange={(event) => setAvoid(event.target.value)}
            rows={3}
            placeholder="Horror, sequels, very long films"
            className="min-h-[88px] resize-none rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-[#fff8ef] outline-none focus:border-[#c45c3e]"
          />
        </label>
        <div className="flex flex-col justify-between gap-3">
          <label className="flex flex-col gap-2 text-sm text-white/70">
            Era
            <select
              value={era}
              onChange={(event) => setEra(event.target.value as typeof era)}
              className="rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-[#fff8ef] outline-none focus:border-[#c45c3e]"
            >
              {ERAS.map((option) => (
                <option key={option.value} value={option.value} className="bg-[#141018]">
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <button
            type="submit"
            disabled={loading}
            className="rounded-2xl bg-[#c45c3e] px-4 py-3 text-sm font-medium text-white transition hover:bg-[#d46b4b] disabled:opacity-60"
          >
            {loading ? "Programming the slate..." : "Recommend movies"}
          </button>
        </div>
      </form>

      {error ? (
        <p className="rounded-2xl border border-red-400/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </p>
      ) : null}

      {result ? (
        <section className="space-y-6">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <p className="max-w-3xl text-lg leading-8 text-[#d7c7b0]">{result.summary}</p>
            <span className="font-mono text-xs tracking-[0.2em] text-white/40">
              {result.source === "openai" ? "OPENAI" : "LOCAL CATALOG"}
            </span>
          </div>
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
            {result.recommendations.map((movie, index) => (
              <MovieCard key={`${movie.title}-${movie.year}`} movie={movie} index={index} />
            ))}
          </div>
        </section>
      ) : (
        <p className="text-sm text-white/40">
          Try the sample taste profile, or replace it with your own night in.
        </p>
      )}
    </div>
  );
}
