import json
import math

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from agr.model import Agr
from agr.records import Question
from agr.serve import EvaluateRequest, state_text, to_questions

from conftest import TOKENIZER, backbone_config

CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
DOC = "The parcel arrived soaked after two days in the rain."
QS = [Question("wet", "boolean", "Is the parcel wet?", ["no", "yes"]),
      Question("colour", "choice", "Which colour?", ["red", "green", "brown"]),
      Question("many", "choice", "Pick one.", [f"option {i}" for i in range(14)]),
      Question("damage", "score", "How damaged?", ["none", "some", "ruined"])]


@pytest.fixture(scope="module")
def letters(tmp_path_factory):
    root = tmp_path_factory.mktemp("letters")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenizer.chat_template = CHAT
    torch.manual_seed(0)
    AutoModel.from_config(backbone_config("gemma4", len(tokenizer))).save_pretrained(root / "backbone")
    tokenizer.save_pretrained(root / "backbone")
    (root / "config.json").write_text(json.dumps({"readout": "letters", "backbone": "tiny-gemma4",
                                                  "max_state_tokens": 512, "max_question_tokens": 256}))
    return Agr.from_pretrained(root, device="cpu", dtype=torch.float32)


def test_letters_layout_follows_the_chat_turn(letters):
    lay = letters.encode(DOC, QS)
    text = letters.tokenizer.decode(lay.ids[:lay.doc_len] + lay.ids[lay.blocks[1][0]:lay.blocks[1][1]])
    assert text.startswith("<|im_start|>user\nContext:\n" + DOC)
    assert text.endswith("\n\nQuestion: Which colour?\nOptions:\n(A) red\n(B) green\n(C) brown<|im_end|>\n<|im_start|>assistant\nAnswer: (")
    score = letters.tokenizer.decode(lay.ids[lay.blocks[3][0]:lay.blocks[3][1]])
    assert "(A) 0: none\n(B) 1: some\n(C) 2: ruined" in score
    wide = letters.tokenizer.decode(lay.ids[lay.blocks[2][0]:lay.blocks[2][1]])
    assert "\n(A) option 0\n(B) option 1" in wide and "\n(N) option 13" in wide


def test_letters_answers_ignore_the_other_questions(letters):
    together = letters.probs(DOC, QS)
    for q, p in zip(QS, together):
        alone = letters.probs(DOC, [q])[0]
        assert p.shape == (len(q.options),) and float(p.sum()) == pytest.approx(1.0, abs=1e-5)
        assert torch.allclose(p, alone, atol=1e-5)


def test_temperature_follows_the_option_count(letters):
    plain = letters.probs(DOC, QS)
    letters.config["temperature_by_options"] = {"a": 3.0, "b": -0.5, "min": 0.05}
    try:
        warm = letters.probs(DOC, QS)
    finally:
        del letters.config["temperature_by_options"]
    for q, p, w in zip(QS, plain, warm):
        t = max(0.05, 3.0 - 0.5 * math.log(len(q.options)))
        assert torch.allclose(w, (p.log() / t).softmax(-1), atol=1e-4)


def test_order_averaging_reads_each_order_once(letters):
    asked = [QS[1], Question("rev", "choice", "Which colour?", ["brown", "green", "red"])]
    forward, backward = letters.probs(DOC, asked)
    letters.config["order_averaging"] = True
    try:
        mean = letters.probs(DOC, [QS[1]])[0]
        assert len(letters.encode(DOC, QS).blocks) == 2 * len(QS) - 1  # a score question is read once
    finally:
        del letters.config["order_averaging"]
    assert torch.allclose(mean, (forward + backward.flip(0)) / 2, atol=1e-5)


def test_letters_requests_render_like_the_layout_expects():
    req = EvaluateRequest(state={"items": list(range(9)), "ok": True}, questions={
        "wet": {"type": "noul", "instructions": "Is it wet?", "criteria": {"true": "soaked", "false": "dry"}},
        "pick": {"type": "choice", "instructions": {"ask": "which"}, "criteria": {"a": {"x": 1}, "b": None}}})
    wet, pick = to_questions(req, letters=True)
    assert wet.options == ["no: dry", "yes: soaked"] and wet.instructions == "Is it wet?"
    assert pick.instructions == '{"ask": "which"}' and pick.options == ['a: {"x": 1}', "b"]
    assert state_text(req.state, letters=True).startswith('{"items": [{"_index": 0, "value": 0}, {"_index": 1')
    old_wet, _ = to_questions(req)
    assert old_wet.options == ["no", "yes"] and "true: soaked" in old_wet.instructions
    assert state_text(req.state) == json.dumps(req.state, indent=2)
