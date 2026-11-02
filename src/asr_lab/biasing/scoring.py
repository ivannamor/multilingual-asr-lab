"""Whether the entities of a reference were written, and whether a list made a model invent them.

An entity counts as written only if all its words are aligned as correct with the reference, in its
own position: a name written elsewhere in the utterance does not count. Global WER hides this, since
names are few and one wrong name costs only as much as any other word.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from asr_lab.evaluation.metrics import EditOp, ErrorCounts, align


def find_phrase(words: Sequence[str], phrase: Sequence[str]) -> list[int]:
    """The positions where a phrase starts in a sequence of words.

    >>> find_phrase("a b a b c".split(), "a b".split())
    [0, 2]
    >>> find_phrase("a b".split(), [])
    []
    """
    if not phrase:
        return []
    return [
        i
        for i in range(len(words) - len(phrase) + 1)
        if list(words[i : i + len(phrase)]) == list(phrase)
    ]


def entity_counts(
    reference: Sequence[str], hypothesis: Sequence[str], entities: Iterable[Sequence[str]]
) -> ErrorCounts:
    """Occurrences of the entities in a reference, as written (hits) or not (substitutions).

    Longer entities go first and keep their words, so that "unione" inside "unione europea" is not
    counted twice. Counting misses as substitutions makes the error rate 1 - recall, so the
    bootstrap of error rates applies to these counts.

    >>> ref, hyp = "la unione europea e roma".split(), "la unione europa e roma".split()
    >>> entity_counts(ref, hyp, [["unione", "europea"], ["unione"], ["roma"]])
    ErrorCounts(hits=1, substitutions=1, deletions=0, insertions=0)
    """
    written = _written_words(reference, hypothesis)
    used: set[int] = set()
    hits = misses = 0
    for entity in sorted({tuple(e) for e in entities}, key=lambda e: (-len(e), e)):
        for start in find_phrase(reference, entity):
            span = range(start, start + len(entity))
            if used.isdisjoint(span):
                used.update(span)
                if all(written[i] for i in span):
                    hits += 1
                else:
                    misses += 1
    return ErrorCounts(hits=hits, substitutions=misses)


def invented_entities(
    reference: Sequence[str], hypothesis: Sequence[str], given: Iterable[Sequence[str]]
) -> int:
    """How many times the hypothesis writes an entity of a list more often than the reference.

    >>> invented_entities("a b".split(), "a b roma roma".split(), [["roma"], ["a"]])
    2
    """
    return sum(
        max(0, len(find_phrase(hypothesis, entity)) - len(find_phrase(reference, entity)))
        for entity in {tuple(e) for e in given}
    )


def _written_words(reference: Sequence[str], hypothesis: Sequence[str]) -> list[bool]:
    """For each reference word, whether the alignment marks it as correct."""
    return [
        pair.op is EditOp.MATCH for pair in align(reference, hypothesis) if pair.ref is not None
    ]
