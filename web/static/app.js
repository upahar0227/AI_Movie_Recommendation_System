const gate = document.querySelector("#gate");
const app = document.querySelector("#app");
const authForm = document.querySelector("#auth-form");
const authError = document.querySelector("#auth-error");
const authSubmit = document.querySelector("#auth-submit");
const genreStep = document.querySelector("#genre-step");
const recs = document.querySelector("#recs");
const genreList = document.querySelector("#genre-list");
const genreError = document.querySelector("#genre-error");
const movieGrid = document.querySelector("#movie-grid");
const historyList = document.querySelector("#history");

let mode = "register";
const selectedGenres = new Set();

function show(user) {
  gate.classList.add("hidden");
  app.classList.remove("hidden");
  document.querySelector("#hello").textContent = user.isNew
    ? `Welcome, ${user.username}`
    : `Welcome back, ${user.username}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || "Something went wrong.");
  }
  return data;
}

function renderMovies(payload) {
  genreStep.classList.add("hidden");
  recs.classList.remove("hidden");
  document.querySelector("#rec-heading").textContent = payload.heading;
  movieGrid.replaceChildren();
  for (const movie of payload.movies) {
    const card = document.createElement("article");
    card.className = "card";
    const title = document.createElement("h3");
    title.textContent = movie.title;
    const meta = document.createElement("p");
    meta.className = "meta";
    const bits = [];
    if (movie.year) bits.push(String(movie.year));
    if (movie.genres?.length) bits.push(movie.genres.join(" · "));
    if (movie.communityRating) bits.push(`${movie.communityRating} avg`);
    meta.textContent = bits.join("  ·  ");
    const reason = document.createElement("p");
    reason.className = "reason";
    reason.textContent = movie.reason || "";
    const stars = document.createElement("div");
    stars.className = "stars";
    for (let value = 1; value <= 5; value += 1) {
      const star = document.createElement("button");
      star.type = "button";
      star.className = "star" + (movie.yourRating >= value ? " filled" : "");
      star.textContent = String(value);
      star.addEventListener("click", () => rate(movie.movieId, value));
      stars.append(star);
    }
    card.append(title, meta, reason, stars);
    movieGrid.append(card);
  }
  historyList.replaceChildren();
  if (!payload.history.length) {
    const empty = document.createElement("li");
    empty.textContent = "Rate a film and the next list will follow your taste.";
    historyList.append(empty);
  }
  for (const item of payload.history) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = item.title;
    const score = document.createElement("strong");
    score.textContent = String(item.yourRating);
    li.append(name, score);
    historyList.append(li);
  }
}

async function loadRecommendations() {
  const payload = await api("/api/recommendations");
  if (payload.mode === "ask-genres") {
    recs.classList.add("hidden");
    genreStep.classList.remove("hidden");
    await loadGenres();
    return;
  }
  renderMovies(payload);
}

async function loadGenres() {
  if (genreList.childElementCount) return;
  const data = await api("/api/genres");
  for (const genre of data.genres) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "genre";
    button.textContent = genre;
    button.addEventListener("click", () => {
      if (selectedGenres.has(genre)) {
        selectedGenres.delete(genre);
        button.classList.remove("selected");
      } else {
        selectedGenres.add(genre);
        button.classList.add("selected");
      }
    });
    genreList.append(button);
  }
}

async function rate(movieId, rating) {
  try {
    const payload = await api("/api/rate", {
      method: "POST",
      body: JSON.stringify({ movieId, rating }),
    });
    const me = await api("/api/me");
    document.querySelector("#hello").textContent = me.user.isNew
      ? `Welcome, ${me.user.username}`
      : `Welcome back, ${me.user.username}`;
    renderMovies(payload);
  } catch (error) {
    document.querySelector("#rec-heading").textContent = error.message;
  }
}

document.querySelector("#tab-register").addEventListener("click", () => {
  mode = "register";
  authSubmit.textContent = "Create account";
  document.querySelector("#tab-register").classList.add("active");
  document.querySelector("#tab-login").classList.remove("active");
});

document.querySelector("#tab-login").addEventListener("click", () => {
  mode = "login";
  authSubmit.textContent = "Sign in";
  document.querySelector("#tab-login").classList.add("active");
  document.querySelector("#tab-register").classList.remove("active");
});

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authError.textContent = "";
  const body = JSON.stringify({
    username: document.querySelector("#username").value,
    password: document.querySelector("#password").value,
  });
  try {
    const data = await api(mode === "register" ? "/api/register" : "/api/login", {
      method: "POST",
      body,
    });
    show(data.user);
    await loadRecommendations();
  } catch (error) {
    authError.textContent = error.message;
  }
});

document.querySelector("#save-genres").addEventListener("click", async () => {
  genreError.textContent = "";
  try {
    await api("/api/genres", {
      method: "POST",
      body: JSON.stringify({ genres: [...selectedGenres] }),
    });
    await loadRecommendations();
  } catch (error) {
    genreError.textContent = error.message;
  }
});

document.querySelector("#logout").addEventListener("click", async () => {
  await api("/api/logout", { method: "POST", body: "{}" });
  location.reload();
});

api("/api/me")
  .then(async (data) => {
    show(data.user);
    await loadRecommendations();
  })
  .catch(() => {
    gate.classList.remove("hidden");
  });
