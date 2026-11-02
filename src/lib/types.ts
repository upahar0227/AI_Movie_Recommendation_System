export type RecommendRequest = {
  mood: string;
  favorites: string;
  avoid: string;
  era: "any" | "classic" | "2000s" | "recent";
};

export type MovieRecommendation = {
  title: string;
  year: number;
  genres: string[];
  vibe: string;
  why: string;
  match: number;
};

export type RecommendResponse = {
  source: "openai" | "catalog";
  summary: string;
  recommendations: MovieRecommendation[];
};
