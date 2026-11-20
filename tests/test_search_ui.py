"""The search box stays closed until the user opens it, and the keyword is not kept in the page."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_search_is_hidden_until_opened() -> None:
    html = _text("frontend/index.html")
    panel = html.split('id="search-panel"', 1)[1].split(">", 1)[0]
    assert "hidden" in panel
    assert 'aria-label="Search"' in html
    assert "Clear search history" in html
    assert 'id="search-history"' not in html
    assert "Search history" not in html
    script = _text("frontend/app.js")
    assert "function closeSearch" in script
    assert 'input.value = ""' in script
    assert "localStorage" not in script
    assert 'sessionStorage.setItem("q"' not in script
    assert "sessionStorage.setItem('q'" not in script
