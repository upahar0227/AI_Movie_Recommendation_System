from __future__ import annotations

import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

_STEMMER = PorterStemmer()
_STOP = None
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def stopword_set() -> set[str]:
    global _STOP
    if _STOP is None:
        _STOP = set(stopwords.words("english"))
    return _STOP


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", str(text).lower())
    stops = stopword_set()
    return [_STEMMER.stem(tok) for tok in tokens if tok not in stops and len(tok) > 2]


def parse_year(title: str) -> int | None:
    match = _YEAR_RE.search(str(title))
    return int(match.group(1)) if match else None


def parse_genres(genre_str: str) -> list[str]:
    if not genre_str or genre_str == "(no genres listed)":
        return []
    return [part.strip() for part in str(genre_str).split("|") if part.strip()]


def clean_title(title: str) -> str:
    return _YEAR_RE.sub("", str(title)).strip()
