"""The named entities of a text, and the lists of them that are given to a model.

Entities come from a named-entity recognizer on the references, because capital letters do not work
across languages: in German every noun has one. The recognizer makes mistakes, so its output passes
a filter and stays in a committed file: the same lists are used in every run, and scoring does not
need the recognizer.

A list is one of three kinds:

- **oracle**: the entities of the utterance itself, the most that a list can do. It uses the
  reference, so it is an upper bound and not something a product could have;
- **domain**: the commonest entities of other text from the domain, the realistic case;
- **distractors**: the entities of another utterance, to measure the harm of a wrong list.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_LEADING_WORDS = frozenset(
    ["il", "lo", "la", "i", "gli", "le", "the", "der", "die", "das", "den", "dem", "des"]
)
_APOSTROPHES = ("'", "\u2019")


def clean_entity(text: str) -> str | None:
    """An entity without leading articles and possessives, or ``None`` if it is not like a name.

    >>> clean_entity("the Indian Ocean"), clean_entity("North Korea's"), clean_entity("truffe")
    ('Indian Ocean', 'North Korea', None)
    >>> clean_entity("Paesi dell'") is None
    True
    """
    words = " ".join(text.split()).split()
    while words and words[0].lower() in _LEADING_WORDS:
        words = words[1:]
    if not words or words[-1].endswith(_APOSTROPHES) or not words[0][:1].isupper():
        return None
    for apostrophe in _APOSTROPHES:
        words[-1] = words[-1].removesuffix(f"{apostrophe}s")
    return " ".join(words)


def extract_entities(
    texts: Iterable[str], model_name: str, labels: Iterable[str]
) -> list[list[str]]:
    """The entities of each text, in order and with repetitions, with a spaCy model.

    A single word at the start of a sentence is dropped: its capital letter says nothing, and it is
    where the models mistake ordinary words for names.
    """
    import spacy  # an optional extra: the lists are committed, and scoring does not need it

    keep = frozenset(labels)
    nlp = spacy.load(model_name, exclude=["lemmatizer", "attribute_ruler", "morphologizer"])
    results = []
    for doc in nlp.pipe(texts, batch_size=32):
        found = []
        for entity in doc.ents:
            single_at_start = len(entity) == 1 and entity.start == entity.sent.start
            cleaned = clean_entity(entity.text)
            if entity.label_ in keep and cleaned and not single_at_start:
                found.append(cleaned)
        results.append(found)
    return results


def domain_list(entities: Iterable[Iterable[str]]) -> list[dict[str, Any]]:
    """Entities of a text from commonest to rarest, whatever their case, in their commonest form.

    >>> domain_list([["Europa", "UE"], ["EUROPA", "Europa"]])
    [{'entity': 'Europa', 'count': 3}, {'entity': 'UE', 'count': 1}]
    """
    forms: dict[str, Counter[str]] = {}
    for found in entities:
        for entity in found:
            forms.setdefault(entity.casefold(), Counter())[entity] += 1
    rows = [
        {"entity": sorted(form.items(), key=lambda item: (-item[1], item[0]))[0][0],
         "count": sum(form.values())}
        for form in forms.values()
    ]  # fmt: skip
    return sorted(rows, key=lambda row: (-row["count"], row["entity"]))


def distractors(entities: Mapping[str, Sequence[str]], seed: int) -> dict[str, list[str]]:
    """For each utterance, the entities of another one, sharing none with it.

    The donor is the next suitable utterance in a shuffled order, so that a list is not always given
    by the same one and the choice is reproducible.

    >>> found = {"a": ["Roma"], "b": ["Parigi"], "c": []}
    >>> distractors(found, 0)["c"] in (["Roma"], ["Parigi"])
    True
    >>> distractors(found, 0)["a"]
    ['Parigi']
    """
    donors = sorted(key for key, value in entities.items() if value)
    random.Random(seed).shuffle(donors)
    result = {}
    for key in entities:
        own = {entity.casefold() for entity in entities[key]}
        start = donors.index(key) + 1 if key in donors else 0
        rotated = donors[start:] + donors[:start]
        donor = next(
            (d for d in rotated if d != key and not own & {e.casefold() for e in entities[d]}),
            None,
        )
        result[key] = list(dict.fromkeys(entities[donor])) if donor else []
    return result


def prompt_text(entities: Sequence[str]) -> str | None:
    """The list as Whisper's initial prompt: names and commas, with no words in any one language.

    >>> prompt_text(["Roma", "Unione europea"])
    'Roma, Unione europea.'
    >>> prompt_text([]) is None
    True
    """
    return ", ".join(entities) + "." if entities else None
