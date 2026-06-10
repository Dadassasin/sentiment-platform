from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from nltk.corpus import stopwords as nltk_stopwords

try:
    import pymorphy3
except ImportError:
    pymorphy3 = None


NEGATION_WORDS = {
    "без",
    "не",
    "нет",
    "ни",
    "никогда",
}

RUSSIAN_STOP_WORDS = set(nltk_stopwords.words("russian")) - NEGATION_WORDS


@dataclass(slots=True)
class PreprocessingOptions:
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stop_words: bool = True
    lemmatize: bool = True


class TextPreprocessor:
    def __init__(self, options: PreprocessingOptions | None = None) -> None:
        self.options = options or PreprocessingOptions()
        self._morph = pymorphy3.MorphAnalyzer() if pymorphy3 is not None else None

    def process(self, text: object) -> str:
        value = "" if text is None else str(text)
        if not any(
            (
                self.options.lowercase,
                self.options.remove_punctuation,
                self.options.remove_stop_words,
                self.options.lemmatize,
            )
        ):
            return value

        if self.options.lowercase:
            value = value.lower()

        if self.options.remove_punctuation:
            value = re.sub(r"[^0-9a-zа-яё\s-]", " ", value, flags=re.IGNORECASE)

        tokens = re.findall(r"[0-9a-zа-яё-]+", value, flags=re.IGNORECASE)
        if self.options.remove_stop_words:
            tokens = [token for token in tokens if token not in RUSSIAN_STOP_WORDS]

        if self.options.lemmatize and self._morph is not None:
            tokens = [self._lemma(token) for token in tokens]

        return " ".join(tokens)

    @lru_cache(maxsize=50_000)
    def _lemma(self, token: str) -> str:
        if self._morph is None:
            return token
        return self._morph.parse(token)[0].normal_form

    def process_many(self, texts: list[object]) -> list[str]:
        return [self.process(text) for text in texts]
