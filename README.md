# ReelMind

An AI movie recommendation app. Describe a mood and a few favorites; the app returns a six-film slate with why each title fits.

## Run it

```bash
npm install
copy .env.example .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## OpenAI (optional, recommended)

Add your key to `.env.local`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Without a key, recommendations come from a built-in catalog so you can still try the app.
