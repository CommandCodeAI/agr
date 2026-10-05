import pytest
import torch
from fastapi.testclient import TestClient

from agr.demo.snake_game import DIRS
from agr.evaluate import mcnemar, request_key, scored, wilson, yes_no
from agr.records import Question
from agr.serve import MAX_BODY_BYTES, Models, create_app, to_answers

REQUEST = {
    "state": "The parcel arrived soaked after two days in the rain.",
    "questions": {
        "wet": {"type": "noul", "instructions": "Is the parcel wet?"},
        "colour": {"type": "choice", "instructions": "Which colour?", "criteria": {"red": None, "green": "like grass", "brown": None}},
        "damage": {"type": "score", "instructions": "How damaged?", "criteria": ["none", "some", "ruined"]},
    },
}


@pytest.fixture
def client(model):
    return TestClient(create_app(Models({"agr": model})))


def test_systemone_answers_in_system_one_shape(client):
    r = client.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200
    body, a = r.json(), r.json()["answers"]
    assert {"model", "answers", "usage", "latency_ms", "truncated"} <= set(body)
    assert a["wet"]["type"] == "noul" and 0 <= a["wet"]["noul"] <= 1
    assert set(a["colour"]["probabilities"]) == {"red", "green", "brown"} and a["colour"]["choice"] in {"red", "green", "brown"}
    assert {"score", "confidence", "legend", "probabilities"} <= set(a["damage"])


def test_evaluate_takes_the_gateway_spelling(client):
    boolean = {**REQUEST, "questions": {"wet": {"type": "boolean", "instructions": "Is the parcel wet?"}}}
    a = client.post("/v1/evaluate", json=boolean).json()["answers"]["wet"]
    b = client.post("/v1/systemone", json=REQUEST).json()["answers"]["wet"]
    assert a["type"] == "boolean" and a["probability"] == pytest.approx(b["noul"], abs=1e-4)


def test_confidence_follows_system_one():
    # choice: (0.4 - 1/4) / (1 - 1/4) = 0.2.  score: mode 0, E|level - 0| = 0.4, uniform spread over
    # three levels = 2/3, so 1 - 0.4 / (2/3) = 0.4.
    qs = [Question("c", "choice", "x", ["a", "b", "c", "d"]), Question("s", "score", "x", ["0", "1", "2"])]
    out = to_answers(qs, [torch.tensor([0.4, 0.4, 0.1, 0.1]), torch.tensor([0.7, 0.2, 0.1])])
    assert out["c"]["confidence"] == pytest.approx(0.2, abs=1e-4)
    assert out["s"]["confidence"] == pytest.approx(0.4, abs=1e-4)


def test_bad_requests_are_refused(client):
    bad_type = {**REQUEST, "questions": {"q": {"type": "yesno", "instructions": "?"}}}
    one_option = {**REQUEST, "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"only": None}}}}
    assert client.post("/v1/systemone", json=bad_type).status_code == 422
    assert client.post("/v1/systemone", json=one_option).status_code == 422
    assert client.post("/v1/systemone", json={**REQUEST, "model": "nope"}).status_code == 404


def test_oversized_bodies_are_refused_even_without_a_length(client):
    def chunked():
        yield b'{"state": "'
        for _ in range(MAX_BODY_BYTES // 65536 + 2):
            yield b"x" * 65536
    r = client.post("/v1/systemone", content=chunked(), headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert client.post("/v1/systemone", content=b"x" * (MAX_BODY_BYTES + 1)).status_code == 413


def test_a_checkpoint_can_set_its_request_limit(model):
    model.config["max_request_tokens"] = 8
    try:
        r = TestClient(create_app(Models({"agr": model}))).post("/v1/systemone", json=REQUEST)
        assert r.status_code == 422 and "the limit is 8" in r.text
    finally:
        del model.config["max_request_tokens"]


def test_api_key_is_required_when_set(model):
    c = TestClient(create_app(Models({"agr": model}), api_key="secret"))
    assert c.get("/healthz").status_code == 200
    assert c.post("/v1/systemone", json=REQUEST).status_code == 401
    assert c.post("/v1/systemone", json=REQUEST, headers={"authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/v1/systemone", json=REQUEST, headers={"authorization": "Bearer secret"}).status_code == 200


def test_snake_plays(client):
    frames = client.get("/demo/snake/frames", params={"session": "test-session", "n": 2}).json()["frames"]
    assert len(frames) == 2 and all(f["move"] in DIRS for f in frames)


def test_comparison_statistics():
    lo, hi = wilson(80, 100)
    assert lo == pytest.approx(0.711, abs=0.001) and hi == pytest.approx(0.867, abs=0.001)
    assert mcnemar(0, 0) == 1.0 and mcnemar(10, 0) == pytest.approx(2 / 1024)


def test_yes_no_labels_are_read_strictly():
    assert [yes_no(x) for x in (True, False, 1, 0, "true", "False", " false ")] == [1, 0, 1, 0, 1, 0, 0]
    assert scored({"type": "noul", "label": "false"}, {"noul": 0.2})[1] == 0
    for bad in ("maybe", "no", 2, None, ""):
        with pytest.raises(ValueError):
            yes_no(bad)


def test_runs_pair_on_content_not_position():
    a = request_key("doc one", {"q": {"type": "noul", "instructions": "?"}})
    b = request_key("doc two", {"q": {"type": "noul", "instructions": "?"}})
    assert a != b and a == request_key("doc one", {"q": {"instructions": "?", "type": "noul"}})
