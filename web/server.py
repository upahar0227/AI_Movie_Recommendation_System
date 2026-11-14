from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.engine import get_catalog  # noqa: E402
from web.store import (  # noqa: E402
    authenticate,
    create_user,
    get_user,
    init_db,
    list_interactions,
    read_session,
    save_rating,
    set_genres,
    sign_session,
)

STATIC = Path(__file__).resolve().parent / "static"
COOKIE = "reel_session"

from src.app.settings import APP_NAME  # noqa: E402

app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()
    get_catalog()


def _current_user(request: Request) -> dict | None:
    user_id = read_session(request.cookies.get(COOKIE))
    if user_id is None:
        return None
    return get_user(user_id)


def _require_user(request: Request) -> dict:
    user = _current_user(request)
    if user is None:
        raise PermissionError("Sign in first.")
    return user


def _attach_session(response: JSONResponse, user_id: int) -> JSONResponse:
    response.set_cookie(
        COOKIE,
        sign_session(user_id),
        httponly=True,
        samesite="lax",
        max_age=14 * 24 * 3600,
        path="/",
    )
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/me")
def me(request: Request):
    user = _current_user(request)
    if user is None:
        return JSONResponse({"user": None}, status_code=401)
    return {"user": user}


@app.post("/api/register")
async def register(request: Request):
    body = await request.json()
    try:
        user = create_user(str(body.get("username", "")), str(body.get("password", "")))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return _attach_session(JSONResponse({"user": user}), user["id"])


@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    user = authenticate(str(body.get("username", "")), str(body.get("password", "")))
    if user is None:
        return JSONResponse({"error": "Username or password is wrong."}, status_code=401)
    return _attach_session(JSONResponse({"user": user}), user["id"])


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/genres")
def genres():
    return {"genres": get_catalog().genres}


@app.post("/api/genres")
async def save_genres(request: Request):
    try:
        user = _require_user(request)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)
    body = await request.json()
    catalog = get_catalog()
    chosen = [genre for genre in body.get("genres", []) if genre in catalog.genres]
    try:
        updated = set_genres(user["id"], chosen)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"user": updated}


@app.get("/api/recommendations")
def recommendations(request: Request):
    try:
        user = _require_user(request)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)
    catalog = get_catalog()
    interactions = list_interactions(user["id"])
    rated = {item["movieId"]: item["rating"] for item in interactions}
    if user["isNew"]:
        if not user["genres"]:
            return {
                "mode": "ask-genres",
                "heading": "What do you like to watch?",
                "movies": [],
                "history": [],
            }
        movies = catalog.top_rated_for_genres(user["genres"], exclude=set(rated))
        heading = "Top rated in " + ", ".join(user["genres"])
        mode = "genre"
    else:
        movies = catalog.from_interactions(interactions)
        heading = "Picked from your ratings"
        mode = "history"
    for movie in movies:
        if movie["movieId"] in rated:
            movie["yourRating"] = rated[movie["movieId"]]
    history = []
    for item in interactions:
        card = catalog.movie_card(item["movieId"], yourRating=item["rating"])
        if card:
            history.append(card)
    return {"mode": mode, "heading": heading, "movies": movies, "history": history}


@app.post("/api/rate")
async def rate(request: Request):
    try:
        user = _require_user(request)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)
    body = await request.json()
    try:
        movie_id = int(body.get("movieId"))
        rating = float(body.get("rating"))
    except (TypeError, ValueError):
        return JSONResponse({"error": "Choose a movie and a rating from 1 to 5."}, status_code=400)
    if get_catalog().movie_card(movie_id) is None:
        return JSONResponse({"error": "That movie is not in the catalog."}, status_code=404)
    try:
        save_rating(user["id"], movie_id, rating)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return recommendations(request)


def main() -> None:
    import uvicorn

    uvicorn.run("web.server:app", host="127.0.0.1", port=3000, reload=False)


if __name__ == "__main__":
    main()
