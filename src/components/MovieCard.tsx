import type { MovieRecommendation } from "@/lib/types";

function posterHue(title: string) {
  let hash = 0;
  for (const char of title) hash = (hash * 31 + char.charCodeAt(0)) % 360;
  return hash;
}

export function MovieCard({ movie, index }: { movie: MovieRecommendation; index: number }) {
  const hue = posterHue(movie.title);

  return (
    <article className="group overflow-hidden rounded-3xl border border-white/10 bg-white/5 backdrop-blur-sm transition hover:-translate-y-1 hover:border-white/20">
      <div
        className="relative aspect-[3/4] p-5"
        style={{
          background: `linear-gradient(160deg, hsl(${hue} 38% 18%), hsl(${(hue + 40) % 360} 28% 8%))`,
        }}
      >
        <span className="font-mono text-xs tracking-[0.2em] text-white/60">
          {String(index + 1).padStart(2, "0")}
        </span>
        <div className="absolute inset-x-5 bottom-5">
          <p className="font-serif text-3xl leading-tight text-white">{movie.title}</p>
          <p className="mt-2 text-sm text-white/70">{movie.year}</p>
        </div>
      </div>
      <div className="space-y-3 p-5">
        <div className="flex flex-wrap gap-2">
          {movie.genres.map((genre) => (
            <span
              key={genre}
              className="rounded-full border border-white/10 px-2.5 py-1 text-xs text-white/70"
            >
              {genre}
            </span>
          ))}
          <span className="rounded-full bg-[#c45c3e]/20 px-2.5 py-1 text-xs text-[#f0b29d]">
            {movie.match}% match
          </span>
        </div>
        <p className="text-sm italic text-[#d7c7b0]">{movie.vibe}</p>
        <p className="text-sm leading-6 text-white/75">{movie.why}</p>
      </div>
    </article>
  );
}
