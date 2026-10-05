"""Typed questions.

`options` is the option text the model reads. `keys` (choice only) keeps the caller's option
names so an answer can be reported under them.
"""

from __future__ import annotations

from dataclasses import dataclass

QTYPES: tuple[str, ...] = ("boolean", "choice", "score")

# Boolean questions always score these two options, in this order, so p(yes) is
# always probabilities[1].
BOOLEAN_OPTIONS: tuple[str, str] = ("no", "yes")


@dataclass
class Question:
    qid: str
    type: str
    instructions: str
    options: list[str]
    label: int | None = None
    keys: list[str] | None = None


def boolean(qid: str, instructions: str, label: bool | None = None) -> Question:
    return Question(qid=qid, type="boolean", instructions=instructions, options=list(BOOLEAN_OPTIONS),
                    label=None if label is None else int(bool(label)))


def choice(qid: str, instructions: str, options: list[str], label: int | None = None, keys: list[str] | None = None) -> Question:
    return Question(qid=qid, type="choice", instructions=instructions, options=list(options), label=label, keys=keys)


def score(qid: str, instructions: str, levels: list[str], label: int | None = None) -> Question:
    return Question(qid=qid, type="score", instructions=instructions, options=list(levels), label=label)
