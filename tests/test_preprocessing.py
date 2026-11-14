from __future__ import annotations

import pandas as pd

from src.recsys.preprocessing import director_name, movie_tag_text, stem_text, top_cast


def test_stem_text_removes_stopwords_and_uses_porter() -> None:
    assert stem_text("The connection is running") == "connect run"


def test_top_cast_and_director() -> None:
    cast = '[{"name": "Ada", "order": 1}, {"name": "Bea", "order": 0}, {"name": "Cy", "order": 2}, {"name": "Dee", "order": 3}]'
    crew = '[{"name": "Greta", "job": "Director"}, {"name": "Sam", "job": "Writer"}]'
    assert top_cast(cast) == ["Bea", "Ada", "Cy"]
    assert director_name(crew) == "Greta"
    row = pd.Series(
        {
            "genres_tmdb": '[{"name": "Drama"}]',
            "keywords": '[{"name": "Space"}]',
            "cast": cast,
            "crew": crew,
            "overview": "A connection in space",
        }
    )
    tags = movie_tag_text(row)
    assert "drama" in tags
    assert "space" in tags
    assert "bea" in tags
    assert "greta" in tags
    assert "connect" in tags
