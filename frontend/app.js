let accessToken = "";
let refreshToken = "";
let accessExpiresAt = 0;
let authMode = "register";
let appView = "login";
let onboardingCompleted = false;
const selectedGenres = new Set();

const gate = document.querySelector("#gate");
const app = document.querySelector("#app");
const authError = document.querySelector("#auth-error");

function authHeader() {
  return { Authorization: "Bearer " + accessToken, "Content-Type": "application/json" };
}

async function refreshAccess() {
  if (!refreshToken) return false;
  const response = await fetch("/auth/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
  if (!response.ok) return false;
  const data = await response.json();
  remember(data);
  return true;
}

async function api(path, options = {}) {
  if (accessToken && Date.now() > accessExpiresAt - 5000) {
    const refreshed = await refreshAccess();
    if (!refreshed) {
      showGate();
      throw new Error("Session expired.");
    }
  }
  const response = await fetch(path, { ...options, headers: { ...authHeader(), ...(options.headers || {}) } });
  if (response.status === 401 && refreshToken) {
    const refreshed = await refreshAccess();
    if (refreshed) {
      return api(path, options);
    }
    showGate();
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail) ? detail.map((item) => item.msg).join(" ") : detail || "Request failed.";
    throw new Error(message);
  }
  return data;
}

function remember(data) {
  accessToken = data.access_token;
  refreshToken = data.refresh_token;
  accessExpiresAt = Date.now() + data.expires_in * 1000;
  sessionStorage.setItem("refresh_token", refreshToken);
}

function showGate() {
  accessToken = "";
  refreshToken = "";
  onboardingCompleted = false;
  sessionStorage.removeItem("refresh_token");
  setView("login");
}

function applyBranding() {
  document.querySelectorAll(".site-name").forEach((el) => {
    el.textContent = SITE_NAME;
  });
  const meta = document.querySelector('meta[name="description"]');
  if (meta) meta.content = SITE_NAME + " recommends films from the genres you like and the movies you rate.";
  const icon = document.querySelector("link[rel='icon']");
  if (icon) icon.title = SITE_NAME;
}

function setPage(page) {
  const labels = {
    login: "Sign in",
    onboarding: "Genres",
    home: "Home",
    ratings: "My Ratings",
    watchlist: "Watchlist",
  };
  document.title = labels[page] + " · " + SITE_NAME;
  const heading = document.querySelector("#page-title");
  if (heading) {
    const show = page === "home" || page === "ratings" || page === "watchlist";
    heading.classList.toggle("hidden", !show);
    if (show) heading.textContent = labels[page];
  }
  document.querySelectorAll("#account-nav a[data-page]").forEach((link) => {
    const on = link.dataset.page === page;
    link.classList.toggle("active", on);
    if (on) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

function closeMenu() {
  document.querySelector("#account-nav").classList.remove("is-open");
  document.querySelector("#nav-toggle").setAttribute("aria-expanded", "false");
}

function canonicalHash(hash) {
  if (hash === "#recommended" || hash === "#/recommended") return "#/home";
  if (hash === "#ratings") return "#/ratings";
  if (hash === "#watchlist") return "#/watchlist";
  if (hash === "#onboarding") return "#/onboarding";
  if (hash === "#/home" || hash === "#/ratings" || hash === "#/watchlist" || hash === "#/onboarding") return hash;
  return "#/home";
}

function setView(next) {
  appView = next;
  gate.classList.toggle("hidden", next !== "login");
  app.classList.toggle("hidden", next === "login");
  const nav = document.querySelector("#account-nav");
  const toggle = document.querySelector("#nav-toggle");
  const hideNav = next !== "main";
  nav.hidden = hideNav;
  nav.setAttribute("aria-hidden", hideNav ? "true" : "false");
  toggle.hidden = hideNav;
  if (hideNav) closeMenu();
  document.querySelector("#genre-step").classList.toggle("hidden", next !== "onboarding");
  if (next === "login") setPage("login");
  if (next === "onboarding") setPage("onboarding");
  if (next !== "main") {
    document.querySelector("#browser").classList.add("hidden");
    document.querySelector("#saved").classList.add("hidden");
    document.querySelector("#detail").classList.add("hidden");
  }
}

function poster(item) {
  const box = document.createElement("div");
  box.className = "poster";
  const letter = (item.title || "?").slice(0, 1);
  if (item.poster_url) {
    const img = document.createElement("img");
    img.alt = item.title || "Poster";
    img.src = item.poster_url;
    img.addEventListener("error", () => {
      img.remove();
      box.textContent = letter;
    });
    box.append(img);
  } else {
    box.textContent = letter;
  }
  return box;
}

function titleLink(item) {
  if (!item.link) {
    const title = document.createElement("strong");
    title.textContent = item.title;
    return title;
  }
  const link = document.createElement("a");
  link.addEventListener("click", () => logClick(item.movie_id));
  link.className = "title-link";
  link.href = item.link;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = item.title;
  return link;
}

function details(item) {
  const wrap = document.createElement("div");
  wrap.className = "copy";
  const credits = [item.lead ? "Lead: " + item.lead : "", item.director ? "Director: " + item.director : ""]
    .filter(Boolean)
    .join(" · ");
  if (credits) {
    const line = document.createElement("div");
    line.className = "muted";
    line.textContent = credits;
    wrap.append(line);
  }
  if (item.overview) {
    const overview = document.createElement("p");
    overview.className = "overview";
    overview.textContent = item.overview;
    wrap.append(overview);
  }
  return wrap;
}

const knownRatings = new Map();
let pendingRating = null;
let refreshTimer = 0;

function rememberedRating(item) {
  if (knownRatings.has(item.movie_id)) return knownRatings.get(item.movie_id);
  if (item.user_rating === null || item.user_rating === undefined) return null;
  return Number(item.user_rating);
}

function toast(message) {
  const node = document.querySelector("#toast");
  node.textContent = message;
  node.classList.remove("hidden");
}

function logClick(movieId) {
  if (!accessToken) return;
  api("/interactions", { method: "POST", body: JSON.stringify({ movie_id: movieId, event_type: "click" }) }).catch(() => {});
}

function syncStars(movieId) {
  const value = knownRatings.has(movieId) ? knownRatings.get(movieId) : null;
  document.querySelectorAll(".star-rating[data-movie-id='" + movieId + "']").forEach((group) => {
    group.dataset.value = value === null ? "" : String(value);
    group.setAttribute("aria-valuenow", value === null ? "0" : String(value));
    group.querySelectorAll("button.half").forEach((button) => {
      button.classList.toggle("on", Number(button.dataset.value) <= value);
      button.setAttribute("aria-checked", Number(button.dataset.value) === value ? "true" : "false");
    });
  });
  document.querySelectorAll(".your-rating-label[data-movie-id='" + movieId + "']").forEach((label) => {
    label.textContent = value ? "Your rating " + value : "Rate this movie";
  });
}

function scheduleRefresh(ratedValue) {
  const updating = document.querySelector("#updating");
  updating.classList.remove("hidden");
  window.clearTimeout(refreshTimer);
  refreshTimer = window.setTimeout(async () => {
    try {
      await loadRecommendations({ stay: true });
      updating.classList.add("hidden");
      if (ratedValue === null) toast("Rating cleared. Recommendations updated.");
      else toast("Rated " + ratedValue + " stars. Recommendations updated.");
    } catch (error) {
      updating.classList.add("hidden");
      toast(error.message);
    }
  }, 500);
}

async function commitRating(movieId, value) {
  if (!accessToken) {
    pendingRating = { movieId, value };
    showGate();
    toast("Sign in to save this rating.");
    return;
  }
  const previous = knownRatings.has(movieId) ? knownRatings.get(movieId) : null;
  const next = value === previous ? null : value;
  knownRatings.set(movieId, next);
  syncStars(movieId);
  try {
    if (next === null) await api("/ratings/" + movieId, { method: "DELETE" });
    else await api("/ratings", { method: "POST", body: JSON.stringify({ movie_id: movieId, rating: next }) });
    scheduleRefresh(next);
  } catch (error) {
    knownRatings.set(movieId, previous);
    syncStars(movieId);
    toast(error.message);
  }
}

function starRating(item) {
  const group = document.createElement("div");
  group.className = "star-rating";
  group.dataset.movieId = String(item.movie_id);
  group.tabIndex = 0;
  group.setAttribute("role", "slider");
  group.setAttribute("aria-label", "Rating for " + item.title);
  group.setAttribute("aria-valuemin", "0");
  group.setAttribute("aria-valuemax", "5");
  group.setAttribute("aria-valuenow", "0");
  const current = rememberedRating(item);
  if (!knownRatings.has(item.movie_id)) knownRatings.set(item.movie_id, current);
  for (let star = 1; star <= 5; star += 1) {
    const wrap = document.createElement("span");
    wrap.className = "star";
    for (const value of [star - 0.5, star]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "half";
      button.dataset.value = String(value);
      button.setAttribute("role", "radio");
      button.setAttribute("aria-label", "Rate " + value + " stars");
      button.addEventListener("mouseenter", () => {
        group.querySelectorAll("button.half").forEach((node) => {
          node.classList.toggle("on", Number(node.dataset.value) <= value);
        });
      });
      button.addEventListener("click", () => commitRating(item.movie_id, value));
      wrap.append(button);
    }
    wrap.addEventListener("mouseleave", () => syncStars(item.movie_id));
    group.append(wrap);
  }
  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "ghost clear-rating";
  clear.textContent = "Clear";
  clear.addEventListener("click", () => {
    const current = knownRatings.get(item.movie_id);
    if (current) commitRating(item.movie_id, current);
  });
  group.addEventListener("keydown", (event) => {
    const now = knownRatings.get(item.movie_id) || 0;
    if (event.key === "ArrowRight") {
      event.preventDefault();
      commitRating(item.movie_id, Math.min(5, now + 0.5));
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      commitRating(item.movie_id, Math.max(0.5, now - 0.5));
    }
  });
  const holder = document.createElement("div");
  holder.className = "rating-controls";
  const label = document.createElement("span");
  label.className = "your-rating-label";
  label.dataset.movieId = String(item.movie_id);
  label.textContent = current ? "Your rating " + current : "Rate this movie";
  holder.append(label, group, clear);
  syncStars(item.movie_id);
  return holder;
}

function communityBadge(item) {
  const node = document.createElement("p");
  node.className = "community-rating";
  if (item.avg_rating === null || item.avg_rating === undefined) {
    node.classList.add("muted");
    node.textContent = "No rating yet";
    return node;
  }
  const count = Number(item.rating_count);
  const compact = count >= 1000 ? (count / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(count);
  node.textContent = "★ " + item.avg_rating + " / " + (item.rating_scale || 10) + " (" + compact + ")";
  const sourceName = item.rating_source === "movielens" ? "MovieLens ratings" : "TMDB votes";
  node.title = item.avg_rating + " from " + count.toLocaleString() + " " + sourceName;
  return node;
}

function sourceBreakdown(item) {
  const node = document.createElement("p");
  node.className = "muted breakdown";
  const parts = [];
  if (item.tmdb_vote_average !== null && item.tmdb_vote_average !== undefined) {
    parts.push("TMDB " + item.tmdb_vote_average + " / 10 from " + Number(item.tmdb_vote_count).toLocaleString() + " votes");
  }
  if (item.movielens_average !== null && item.movielens_average !== undefined) {
    parts.push("MovieLens " + item.movielens_average + " / 10 from " + Number(item.movielens_count).toLocaleString() + " ratings");
  }
  node.textContent = parts.length ? parts.join(" · ") : "No community breakdown";
  return node;
}

function movieCard(item, options = {}) {
  const row = document.createElement("article");
  row.className = "card";
  const text = document.createElement("div");
  text.className = "copy";
  const why = document.createElement("div");
  why.className = "muted";
  why.textContent = item.reason || item.explanation || item.genres || "";
  const more = document.createElement("button");
  more.type = "button";
  more.className = "ghost";
  more.textContent = "Similar";
  more.addEventListener("click", () => showSimilar(item));
  text.append(titleLink(item), communityBadge(item), details(item), why, more);
  if (options.detail) text.append(sourceBreakdown(item));
  const save = document.createElement("button");
  save.type = "button";
  save.className = "ghost";
  save.textContent = options.removeLabel || "Watchlist";
  save.addEventListener("click", async () => {
    try {
      if (options.onRemove) await options.onRemove();
      else {
        await api("/watchlist/" + item.movie_id, { method: "POST" });
        toast(item.title + " was added to your watchlist.");
      }
    } catch (error) {
      toast(error.message);
    }
  });
  row.append(poster(item), text, starRating(item), save);
  return row;
}

async function showSimilar(item) {
  const box = document.querySelector("#detail");
  box.classList.remove("hidden");
  box.replaceChildren();
  const heading = document.createElement("h2");
  heading.textContent = "Similar to " + item.title;
  box.append(heading, movieCard(item, { detail: true }));
  try {
    logClick(item.movie_id);
    const payload = await api("/recommendations/similar/" + item.movie_id + "?n=6");
    for (const related of payload.results) box.append(movieCard(related));
  } catch (error) {
    box.append(error.message);
  }
}

function showPanel(name) {
  document.querySelector("#genre-step").classList.toggle("hidden", name !== "genres");
  document.querySelector("#browser").classList.toggle("hidden", name !== "recs");
  document.querySelector("#saved").classList.toggle("hidden", name !== "saved");
}

async function loadGenres() {
  const data = await api("/genres");
  const box = document.querySelector("#genre-list");
  box.replaceChildren();
  selectedGenres.clear();
  for (const genre of data.genres) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost";
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
    box.append(button);
  }
}

async function loadRecommendations(options = {}) {
  const payload = await api("/recommendations/me?n=10");
  const notice = document.querySelector("#notice");
  if (!onboardingCompleted || payload.mode === "ask_genres") {
    notice.textContent = "Pick the genres you like to see top-rated films.";
    if (!options.stay) {
      setView("onboarding");
      await loadGenres();
    }
    return;
  }
  if (!options.stay) {
    setPage("home");
    setView("main");
    showPanel("recs");
  }
  if (payload.mode === "genre") {
    notice.textContent = "These are the highest-rated films in the genres you picked.";
  } else {
    notice.textContent = "Recommendations are based on your ratings.";
  }
  const results = document.querySelector("#results");
  results.replaceChildren();
  document.querySelector("#heading").textContent = payload.mode === "genre" ? "Top rated in your genres" : "Recommended for you";
  for (const item of payload.results) {
    knownRatings.set(item.movie_id, item.user_rating === null || item.user_rating === undefined ? null : Number(item.user_rating));
    results.append(movieCard(item));
  }
  await loadRecent();
  try {
    await renderYourGenres();
  } catch (error) {
    document.querySelector("#notice").textContent = error.message;
  }
}

async function loadRecent() {
  const box = document.querySelector("#recent");
  if (!box) return;
  box.replaceChildren();
  const payload = await api("/ratings/me");
  if (!payload.ratings.length) {
    box.textContent = "Movies you rate will show up here and leave the recommendation list.";
    return;
  }
  for (const item of payload.ratings.slice(0, 4)) box.append(movieCard(item));
}

async function search(value) {
  const hits = document.querySelector("#hits");
  hits.replaceChildren();
  if (value.trim().length < 1) return;
  const payload = await api("/movies/search?q=" + encodeURIComponent(value));
  for (const item of payload.results) {
    if (item.user_rating !== undefined) knownRatings.set(item.movie_id, item.user_rating);
    hits.append(movieCard(item));
  }
}

async function showList(kind) {
  if (appView !== "main" || !onboardingCompleted) {
    setView("onboarding");
    if (location.hash !== "#/onboarding") history.replaceState(null, "", "#/onboarding");
    await loadGenres();
    return;
  }
  showPanel("saved");
  setPage(kind === "ratings" ? "ratings" : "watchlist");
  const box = document.querySelector("#lists");
  box.replaceChildren();
  try {
    if (kind === "ratings") {
      const payload = await api("/ratings/me");
      if (!payload.ratings.length) {
        box.textContent = "You have not rated any movies yet.";
        return;
      }
      for (const item of payload.ratings) {
        knownRatings.set(item.movie_id, Number(item.rating));
        box.append(movieCard(item, {
          removeLabel: "Remove",
          onRemove: async () => {
            await api("/ratings/" + item.movie_id, { method: "DELETE" });
            knownRatings.set(item.movie_id, null);
            await showList("ratings");
          },
        }));
      }
    } else {
      const payload = await api("/watchlist/me");
      if (!payload.watchlist.length) {
        box.textContent = "Your watchlist is empty.";
        return;
      }
      for (const item of payload.watchlist) {
        if (item.user_rating !== undefined) knownRatings.set(item.movie_id, item.user_rating);
        box.append(movieCard(item, {
          removeLabel: "Remove",
          onRemove: async () => {
            await api("/watchlist/" + item.movie_id, { method: "DELETE" });
            await showList("watch");
          },
        }));
      }
    }
  } catch (error) {
    box.textContent = error.message;
  }
}

document.querySelector("#tab-register").addEventListener("click", () => {
  authMode = "register";
  const username = document.querySelector("#username");
  username.required = true;
  username.classList.remove("hidden");
  document.querySelector("label[for='username']").classList.remove("hidden");
  document.querySelector("#auth-submit").textContent = "Create account";
  document.title = "Create account · " + SITE_NAME;
});
document.querySelector("#tab-login").addEventListener("click", () => {
  authMode = "login";
  const username = document.querySelector("#username");
  username.required = false;
  username.classList.add("hidden");
  document.querySelector("label[for='username']").classList.add("hidden");
  document.querySelector("#auth-submit").textContent = "Sign in";
  setPage("login");
});

document.querySelector("#auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  authError.textContent = "";
  const email = document.querySelector("#email").value.trim();
  const username = document.querySelector("#username").value.trim();
  const password = document.querySelector("#password").value;
  if (authMode === "register" && (username.length < 3 || password.length < 8 || /^\d+$/.test(password))) {
    authError.textContent = "Use a username of 3 or more characters and a password of 8 or more characters that is not only numbers.";
    return;
  }
  try {
    if (authMode === "register") {
      await api("/auth/register", { method: "POST", body: JSON.stringify({ email, username, password }) });
    }
    const body = new URLSearchParams({ username: email, password });
    const response = await fetch("/auth/login", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Email or password is wrong.");
    remember(data);
    const me = await api("/users/me");
    onboardingCompleted = Boolean(me.onboarding_completed);
    document.querySelector("#hello").textContent = "Hello, " + me.username;
    if (pendingRating && onboardingCompleted) {
      const waiting = pendingRating;
      pendingRating = null;
      await commitRating(waiting.movieId, waiting.value);
    }
    history.replaceState(null, "", onboardingCompleted ? "#/home" : "#/onboarding");
    await followHash();
  } catch (error) {
    authError.textContent = error.message;
  }
});

document.querySelector("#find").addEventListener("click", () => search(document.querySelector("#q").value));
document.querySelector("#q").addEventListener("input", (event) => search(event.target.value));
document.querySelector("#save-genres").addEventListener("click", async () => {
  const error = document.querySelector("#genre-error");
  error.textContent = "";
  if (!selectedGenres.size) {
    error.textContent = "Pick at least one genre.";
    return;
  }
  try {
    await api("/genres", { method: "POST", body: JSON.stringify({ genres: [...selectedGenres] }) });
    onboardingCompleted = true;
    history.replaceState(null, "", "#/home");
    await followHash();
  } catch (err) {
    error.textContent = err.message;
  }
});
async function followHash() {
  if (!accessToken) {
    setPage("login");
    return;
  }
  const next = canonicalHash(location.hash);
  if (location.hash !== next) history.replaceState(null, "", next);
  closeMenu();
  if (!onboardingCompleted) {
    if (next !== "#/onboarding") history.replaceState(null, "", "#/onboarding");
    setView("onboarding");
    try {
      await loadGenres();
    } catch (error) {
      document.querySelector("#notice").textContent = error.message;
    }
    return;
  }
  setView("main");
  if (next === "#/ratings") await showList("ratings");
  else if (next === "#/watchlist") await showList("watch");
  else await loadRecommendations();
}

window.addEventListener("hashchange", () => {
  followHash().catch((error) => {
    document.querySelector("#notice").textContent = error.message;
  });
});

document.querySelector("#nav-toggle").addEventListener("click", () => {
  const nav = document.querySelector("#account-nav");
  const open = nav.classList.toggle("is-open");
  document.querySelector("#nav-toggle").setAttribute("aria-expanded", open ? "true" : "false");
});

document.querySelector("#logout").addEventListener("click", async () => {
  await api("/auth/logout", { method: "POST", body: JSON.stringify({ refresh_token: refreshToken }) });
  showGate();
});
document.querySelector("#delete-account").addEventListener("click", async () => {
  if (!window.confirm("Delete your account and all ratings?")) return;
  await api("/users/me", { method: "DELETE" });
  showGate();
});

async function restoreSession() {
  const stored = sessionStorage.getItem("refresh_token");
  if (!stored) {
    setView("login");
    return;
  }
  refreshToken = stored;
  const refreshed = await refreshAccess();
  if (!refreshed) {
    showGate();
    return;
  }
  try {
    const me = await api("/users/me");
    onboardingCompleted = Boolean(me.onboarding_completed);
    document.querySelector("#hello").textContent = "Hello, " + me.username;
    await followHash();
  } catch (error) {
    showGate();
  }
}

let genreEditorKeydown = null;
let genreReturnFocus = null;

async function renderYourGenres() {
  const row = document.querySelector("#genre-chips");
  if (!row || appView !== "main") return;
  const payload = await api("/users/me/genres");
  row.replaceChildren();
  for (const genre of payload.genres) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = genre;
    row.append(chip);
  }
  if (!payload.genres.length) {
    const empty = document.createElement("span");
    empty.className = "muted";
    empty.textContent = "None yet";
    row.append(empty);
  }
  row.dataset.max = String(payload.max || 8);
  row.dataset.allowed = JSON.stringify(payload.allowed || []);
  row.dataset.selected = JSON.stringify(payload.genres || []);
}

function genreEditorItems() {
  return [...document.querySelectorAll("#genre-editor button")];
}

function closeGenreEditor() {
  const editor = document.querySelector("#genre-editor");
  editor.hidden = true;
  if (genreEditorKeydown) document.removeEventListener("keydown", genreEditorKeydown);
  genreEditorKeydown = null;
  if (genreReturnFocus) genreReturnFocus.focus();
}

function openGenreEditor() {
  const editor = document.querySelector("#genre-editor");
  const list = document.querySelector("#genre-editor-list");
  const chips = document.querySelector("#genre-chips");
  const allowed = JSON.parse(chips.dataset.allowed || "[]");
  const selected = new Set(JSON.parse(chips.dataset.selected || "[]"));
  const max = Number(chips.dataset.max || 8);
  const error = document.querySelector("#genre-editor-error");
  error.textContent = "";
  list.replaceChildren();
  function updateCount() {
    document.querySelector("#genre-count").textContent = selected.size + " of " + max + " selected";
  }
  for (const genre of allowed) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost";
    button.textContent = genre;
    button.setAttribute("aria-pressed", selected.has(genre) ? "true" : "false");
    button.setAttribute("aria-label", genre);
    if (selected.has(genre)) button.classList.add("selected");
    button.addEventListener("click", () => {
      if (selected.has(genre)) selected.delete(genre);
      else selected.add(genre);
      button.classList.toggle("selected", selected.has(genre));
      button.setAttribute("aria-pressed", selected.has(genre) ? "true" : "false");
      error.textContent = "";
      updateCount();
    });
    list.append(button);
  }
  updateCount();
  genreReturnFocus = document.activeElement;
  editor.hidden = false;
  const first = list.querySelector("button") || document.querySelector("#genre-cancel");
  first.focus();
  if (genreEditorKeydown) document.removeEventListener("keydown", genreEditorKeydown);
  genreEditorKeydown = (event) => {
    if (editor.hidden) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeGenreEditor();
      return;
    }
    if (event.key !== "Tab") return;
    const items = genreEditorItems();
    if (!items.length) return;
    const firstItem = items[0];
    const lastItem = items[items.length - 1];
    if (event.shiftKey && document.activeElement === firstItem) {
      event.preventDefault();
      lastItem.focus();
    } else if (!event.shiftKey && document.activeElement === lastItem) {
      event.preventDefault();
      firstItem.focus();
    }
  };
  document.addEventListener("keydown", genreEditorKeydown);
  document.querySelector("#genre-save").onclick = async () => {
    error.textContent = "";
    if (!selected.size) {
      error.textContent = "Pick at least one genre.";
      return;
    }
    if (selected.size > max) {
      error.textContent = "Choose at most " + max + " genres.";
      return;
    }
    try {
      await api("/users/me/genres", { method: "PUT", body: JSON.stringify({ genres: [...selected] }) });
      closeGenreEditor();
      const updating = document.querySelector("#updating");
      updating.classList.remove("hidden");
      await loadRecommendations({ stay: true });
      updating.classList.add("hidden");
      toast("Your genres were updated.");
    } catch (err) {
      document.querySelector("#updating").classList.add("hidden");
      error.textContent = err.message;
    }
  };
}

document.querySelector("#edit-genres").addEventListener("click", () => openGenreEditor());
document.querySelector("#genre-cancel").addEventListener("click", () => closeGenreEditor());
document.querySelector("#genre-editor").addEventListener("click", (event) => {
  if (event.target.id === "genre-editor") closeGenreEditor();
});

applyBranding();
restoreSession();
