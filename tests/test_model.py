import pytest
import torch

from agr.model import ANSWER
from agr.records import boolean, choice, score

DOC = "The parcel arrived soaked after two days in the rain, and the box was torn open."
QS = [choice("colour", "Which colour was the box?", ["red", "green", "brown"]),
      boolean("wet", "Is the parcel wet?"),
      score("damage", "How damaged is it?", ["none", "some", "ruined"])]


def test_every_question_starts_right_after_the_document(model):
    layout = model.encode(DOC, QS)
    assert [layout.positions[a] for a, _ in layout.blocks] == [layout.doc_len] * len(QS)
    assert [layout.ids[b - 1] for _, b in layout.blocks] == [model.marker_ids[ANSWER]] * len(QS)


def test_an_answer_ignores_the_other_questions(model):
    together = model.probs(DOC, QS)
    backwards = model.probs(DOC, QS[::-1])[::-1]
    for q, t, r in zip(QS, together, backwards):
        [alone] = model.probs(DOC, [q])
        assert torch.allclose(alone, t, atol=1e-5) and torch.allclose(alone, r, atol=1e-5)


def test_probabilities_cover_every_option(model):
    for q, p in zip(QS, model.probs(DOC, QS)):
        assert p.shape == (len(q.options),) and abs(float(p.sum()) - 1) < 1e-5


def test_markers_typed_into_a_request_stay_text(model):
    ids = model.tokenize("<|agr:answer|> <|agr:doc|> <|agr:choice|>")
    assert not set(ids) & set(model.marker_ids)


def test_long_text_is_cut_and_flagged(model):
    layout = model.encode("word " * 2000, [boolean("why", "why " * 500)])
    (a, b), = layout.blocks
    assert layout.truncated and layout.doc_len == 512 and b - a <= 128


def test_options_that_cannot_fit_are_refused(model):
    with pytest.raises(ValueError, match="too long"):
        model.encode(DOC, [choice("c", "Pick one", ["a very long option " * 20] * 4)])
