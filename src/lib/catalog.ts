import type { MovieRecommendation, RecommendRequest } from "./types";

type CatalogMovie = {
  title: string;
  year: number;
  genres: string[];
  vibe: string;
  keywords: string[];
  why: string;
};

export const CATALOG: CatalogMovie[] = [
  {
    title: "Arrival",
    year: 2016,
    genres: ["Sci-Fi", "Drama"],
    vibe: "Quiet, linguistic, emotionally precise",
    keywords: ["thoughtful", "scifi", "sci-fi", "language", "rainy", "arrival", "her", "interstellar"],
    why: "A patient first-contact story that rewards close attention more than spectacle.",
  },
  {
    title: "Her",
    year: 2013,
    genres: ["Romance", "Sci-Fi", "Drama"],
    vibe: "Tender, near-future loneliness",
    keywords: ["romance", "thoughtful", "lonely", "ai", "her", "cozy"],
    why: "Soft futurism about connection, longing, and how we talk to the people we love.",
  },
  {
    title: "Blade Runner 2049",
    year: 2017,
    genres: ["Sci-Fi", "Mystery"],
    vibe: "Neon melancholy and scale",
    keywords: ["scifi", "sci-fi", "neon", "noir", "blade runner", "dune", "visual"],
    why: "A lavish, lonely detective story if you want atmosphere more than plot speed.",
  },
  {
    title: "The Grand Budapest Hotel",
    year: 2014,
    genres: ["Comedy", "Adventure"],
    vibe: "Confectionary, precise, bittersweet",
    keywords: ["funny", "whimsical", "colorful", "wes anderson", "cozy"],
    why: "Stylish comedy with a surprisingly aching core if you want something delightful.",
  },
  {
    title: "Moonlight",
    year: 2016,
    genres: ["Drama"],
    vibe: "Intimate, lyrical, unforgettable",
    keywords: ["drama", "thoughtful", "quiet", "coming of age", "rainy"],
    why: "A small, luminous portrait of identity and tenderness rather than a plot engine.",
  },
  {
    title: "Mad Max: Fury Road",
    year: 2015,
    genres: ["Action", "Adventure"],
    vibe: "Relentless, sun-blasted, operatic",
    keywords: ["action", "intense", "chase", "fun", "loud"],
    why: "If you want momentum and craft, this is practically a moving storyboard.",
  },
  {
    title: "Paddington 2",
    year: 2017,
    genres: ["Family", "Comedy"],
    vibe: "Warm, witty, sincerely kind",
    keywords: ["cozy", "funny", "family", "feel good", "rainy", "comfort"],
    why: "A comfort movie that is also genuinely well-made, not just sweet.",
  },
  {
    title: "Parasite",
    year: 2019,
    genres: ["Thriller", "Drama"],
    vibe: "Sharp, funny, then devastating",
    keywords: ["thriller", "class", "dark comedy", "intense", "bong"],
    why: "A social thriller that keeps changing shape without losing its bite.",
  },
  {
    title: "Everything Everywhere All at Once",
    year: 2022,
    genres: ["Action", "Comedy", "Drama"],
    vibe: "Chaotic, tender, maximalist",
    keywords: ["fun", "family", "multiverse", "weird", "recent"],
    why: "If you want invention plus feeling, this swings harder than almost anything recent.",
  },
  {
    title: "Past Lives",
    year: 2023,
    genres: ["Romance", "Drama"],
    vibe: "Hushed, adult, lingering",
    keywords: ["romance", "quiet", "thoughtful", "recent", "rainy"],
    why: "A modern romance that trusts silences and unfinished conversations.",
  },
  {
    title: "Dune: Part Two",
    year: 2024,
    genres: ["Sci-Fi", "Adventure"],
    vibe: "Mythic, sandy, monumental",
    keywords: ["scifi", "sci-fi", "epic", "dune", "visual", "recent"],
    why: "Grand-scale science fiction for when you want weight and world-building.",
  },
  {
    title: "The Holdovers",
    year: 2023,
    genres: ["Comedy", "Drama"],
    vibe: "Wintry, cranky, eventually warm",
    keywords: ["cozy", "christmas", "school", "comedy", "recent", "comfort"],
    why: "A character piece that feels like a well-worn sweater by the fire.",
  },
  {
    title: "Spirited Away",
    year: 2001,
    genres: ["Animation", "Fantasy"],
    vibe: "Dreamlike, generous, strange",
    keywords: ["animation", "fantasy", "magical", "family", "ghibli"],
    why: "A wonder-story that works for kids and still feels inexhaustible to adults.",
  },
  {
    title: "Before Sunset",
    year: 2004,
    genres: ["Romance", "Drama"],
    vibe: "Walk-and-talk intimacy",
    keywords: ["romance", "talky", "paris", "thoughtful", "2000s"],
    why: "If you like conversations more than set pieces, this is almost a perfect hour and twenty.",
  },
  {
    title: "The Social Network",
    year: 2010,
    genres: ["Drama", "Biography"],
    vibe: "Fast, icy, verbally lethal",
    keywords: ["drama", "tech", "dialogue", "sorkin", "2000s"],
    why: "A propulsion machine of talk and ambition if you want something sharp.",
  },
  {
    title: "Get Out",
    year: 2017,
    genres: ["Horror", "Thriller"],
    vibe: "Satirical, tense, surgical",
    keywords: ["horror", "thriller", "social", "intense"],
    why: "Horror that is also a thesis, without losing the scare craft.",
  },
  {
    title: "Whiplash",
    year: 2014,
    genres: ["Drama", "Music"],
    vibe: "Sweaty, percussive, extreme",
    keywords: ["intense", "music", "ambition", "drama"],
    why: "A pressure-cooker if you want talent, obsession, and no easy comfort.",
  },
  {
    title: "Amélie",
    year: 2001,
    genres: ["Romance", "Comedy"],
    vibe: "Parisian, whimsical, visually sweet",
    keywords: ["romance", "cozy", "whimsical", "colorful", "2000s"],
    why: "A confection of small kindnesses when you want charm without cynicism.",
  },
  {
    title: "The Iron Giant",
    year: 1999,
    genres: ["Animation", "Family", "Sci-Fi"],
    vibe: "Heartfelt, classic, sincere",
    keywords: ["family", "animation", "classic", "feel good", "comfort"],
    why: "A classic about friendship and choice that still lands emotionally.",
  },
  {
    title: "In the Mood for Love",
    year: 2000,
    genres: ["Romance", "Drama"],
    vibe: "Glances, corridors, withheld desire",
    keywords: ["romance", "classic", "quiet", "rainy", "stylish"],
    why: "If you want atmosphere and restraint, almost every frame is a still you could frame.",
  },
];

function tokens(value: string) {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 2);
}

function eraMatches(year: number, era: RecommendRequest["era"]) {
  if (era === "any") return true;
  if (era === "classic") return year < 2000;
  if (era === "2000s") return year >= 2000 && year < 2015;
  return year >= 2015;
}

export function recommendFromCatalog(request: RecommendRequest): {
  summary: string;
  recommendations: MovieRecommendation[];
} {
  const haystack = tokens(`${request.mood} ${request.favorites}`);
  const avoid = tokens(request.avoid);

  const scored = CATALOG.filter((movie) => {
    if (!eraMatches(movie.year, request.era)) return false;
    const blob = `${movie.title} ${movie.genres.join(" ")} ${movie.keywords.join(" ")}`.toLowerCase();
    return !avoid.some((word) => blob.includes(word));
  })
    .map((movie) => {
      const overlap = haystack.filter((token) =>
        movie.keywords.some((keyword) => keyword.includes(token) || token.includes(keyword)),
      ).length;
      const favoriteHit = tokens(request.favorites).some((fav) =>
        movie.title.toLowerCase().includes(fav),
      );
      return {
        movie,
        score: overlap * 12 + (favoriteHit ? -40 : 18) + Math.max(0, 10 - Math.abs(2018 - movie.year) / 4),
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);

  return {
    summary:
      "These picks come from a built-in catalog while your OpenAI key is missing. Add OPENAI_API_KEY to .env.local for richer, more personal AI recommendations.",
    recommendations: scored.map(({ movie, score }) => ({
      title: movie.title,
      year: movie.year,
      genres: movie.genres,
      vibe: movie.vibe,
      why: movie.why,
      match: Math.max(62, Math.min(97, Math.round(70 + score / 4))),
    })),
  };
}
