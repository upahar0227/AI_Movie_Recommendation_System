"""FastAPI service. Artifacts are loaded once at startup."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.recsys.posters import cached_poster
from api.library import apply_user_ratings, rating_map
from src.recsys.search_profile import tie_break_search_results
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.auth import router as auth_router  # noqa: E402
from api.library import router as library_router  # noqa: E402
from api.limiter import limiter  # noqa: E402
from src.app.database import make_engine, make_session_factory  # noqa: E402
from src.app.models import RevokedToken, User  # noqa: E402
from src.app.security import auth_settings, decode_token  # noqa: E402
from src.app.settings import APP_NAME  # noqa: E402
from src.recsys.serving import RecommenderService, load_service  # noqa: E402

logger = logging.getLogger(__name__)
FRONTEND = ROOT / "frontend" / "index.html"


class RatedMovie(BaseModel):
    movie_id: int
    rating: float = Field(ge=0.5, le=5)


class ProfileRequest(BaseModel):
    ratings: list[RatedMovie] = Field(default_factory=list)


def _dump(item) -> dict:
    return item.to_dict()


def create_app(
    service: RecommenderService | None = None,
    *,
    database_url: str | None = None,
    enforce_secret: bool | None = None,
) -> FastAPI:
    app = FastAPI(title=APP_NAME)
    app.state.service = service
    app.state.session_factory = None
    app.state.limiter = limiter
    app.state.database_url = database_url
    app.state.enforce_secret = service is None and database_url is None if enforce_secret is None else enforce_secret
    if database_url is not None:
        app.state.session_factory = make_session_factory(make_engine(database_url))
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    load_dotenv(ROOT / ".env")
    origins = [item.strip() for item in os.environ.get("CORS_ORIGINS", "http://127.0.0.1:8000").split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.on_event("startup")
    def startup() -> None:
        load_dotenv(ROOT / ".env")
        if app.state.enforce_secret:
            auth_settings()
        if app.state.service is None:
            try:
                app.state.service = load_service()
            except FileNotFoundError as exc:
                logger.error("%s", exc)
                raise
        if app.state.database_url is not None or app.state.enforce_secret:
            engine = make_engine(app.state.database_url)
            app.state.session_factory = make_session_factory(engine)

    def recommender() -> RecommenderService:
        current = getattr(app.state, "service", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Models are not loaded. Train them first.")
        return current

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND)

    @app.get("/posters/{imdb_id}")
    def poster(imdb_id: str) -> FileResponse:
        path = cached_poster(imdb_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Poster not available.")
        with path.open("rb") as handle:
            head = handle.read(12)
        if head.startswith(b"\x89PNG"):
            media = "image/png"
        elif head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            media = "image/webp"
        else:
            media = "image/jpeg"
        return FileResponse(path, media_type=media)

    @app.get("/health")
    def health() -> dict:
        current = recommender()
        return {
            "status": "ok",
            "movies": int(current.movies["movieId"].nunique()),
            "reception_method": current.reception_method,
        }

    @app.get("/movies/search")
    def search(request: Request, q: str = Query(min_length=1)) -> dict:
        results = recommender().search(q)
        user, session = _open_user(request)
        if user is not None and session is not None:
            try:
                scores = rating_map(session, user.id)
                results = tie_break_search_results(
                    recommender(),
                    q,
                    results,
                    list(scores.items()),
                )
                apply_user_ratings(results, scores)
            finally:
                session.close()
        return {"results": results}

    @app.get("/recommendations/similar/{movie_id}")
    def similar(movie_id: int, request: Request, n: int = Query(default=10, ge=1, le=50)) -> dict:
        try:
            results = [_dump(item) for item in recommender().similar(movie_id, n)]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _personalise(request, results)
        return {"results": results}

    @app.get("/recommendations/user/{user_id}")
    def for_user(user_id: int, n: int = Query(default=10, ge=1, le=50)) -> dict:
        try:
            results = recommender().for_user(user_id, n)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"results": [_dump(item) for item in results]}

    @app.post("/recommendations/profile")
    def for_profile(body: ProfileRequest, n: int = Query(default=10, ge=1, le=50)) -> dict:
        ratings = [(item.movie_id, item.rating) for item in body.ratings]
        try:
            results = recommender().for_profile(ratings, n)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"results": [_dump(item) for item in results]}

    def _personalise(request: Request, cards: list[dict]) -> None:
        user, session = _open_user(request)
        if user is None or session is None:
            return
        try:
            apply_user_ratings(cards, rating_map(session, user.id))
        finally:
            session.close()

    app.include_router(auth_router)
    app.include_router(library_router)
    app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
    return app


def _open_user(request: Request):
    header = request.headers.get("authorization", "")
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None or not header.lower().startswith("bearer "):
        return None, None
    try:
        payload = decode_token(header.split(" ", 1)[1].strip(), auth_settings())
    except Exception:
        return None, None
    if payload.get("type") != "access":
        return None, None
    session = factory()
    revoked = session.get(RevokedToken, payload.get("jti"))
    user = None if revoked else session.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        session.close()
        return None, None
    return user, session


app = create_app()
