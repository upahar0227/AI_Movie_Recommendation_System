from src.recsys.posters import imdb_tt, imdb_url, poster_path


def test_imdb_id_and_poster_path() -> None:
    assert imdb_tt(114709) == "tt0114709"
    assert imdb_tt(114709.0) == "tt0114709"
    assert imdb_tt(None) is None
    assert poster_path("tt0114709") == "/posters/tt0114709"
    assert poster_path("not-an-id") is None
    assert imdb_url("tt0114709") == "https://www.imdb.com/title/tt0114709/"
    assert imdb_url("not-an-id") is None
