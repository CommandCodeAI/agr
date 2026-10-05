"""HTTP server: `POST /v1/systemone` and `POST /v1/evaluate`, plus the Snake demo.

The request is TypeSafe's System One schema, so their SDK works unchanged. The two routes are the
two spellings Jev is served under: `/v1/systemone` with `noul` (TypeSafe's API) and `/v1/evaluate`
with `boolean` (Vercel AI Gateway). Either route takes either spelling, and a yes/no answer comes
back under the name it was asked with.

    {"model": "agr", "state": <text or JSON>,
     "questions": {"<id>": {"type": "noul" | "boolean", "instructions": ...},
                   "<id>": {"type": "choice", "instructions": ..., "criteria": {"name": "description", ...}},
                   "<id>": {"type": "score",  "instructions": ..., "criteria": ["level 0", "level 1", ...]}}}
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Literal, Union

from fastapi import Request  # module level: endpoint annotations are resolved against module globals
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .model import Agr
from .records import BOOLEAN_OPTIONS, Question

JSONContent = Union[str, dict, list, int, float, bool, None]

# One request is one forward pass over document + questions, so its size bounds GPU memory.
MAX_QUESTIONS = 64
MAX_TOKENS = 12_288
MAX_BODY_BYTES = 1_000_000


class RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class BooleanQuestion(BaseModel):
    type: Literal["noul", "boolean"]
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent
    criteria: dict[str, JSONContent] = Field(min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=255)


class EvaluateRequest(BaseModel):
    state: JSONContent
    model: str = "agr"
    questions: dict[str, Union[BooleanQuestion, ChoiceQuestion, ScoreQuestion]] = Field(min_length=1, max_length=MAX_QUESTIONS)


def _text(v: JSONContent, indent: int | None = 2) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=indent)


def _indexed(v: JSONContent, at_least: int = 8) -> JSONContent:
    """Long arrays carry each element's position, so a question about `items[47]` is a lookup
    rather than a count."""
    if isinstance(v, list):
        if len(v) < at_least:
            return [_indexed(x, at_least) for x in v]
        return [{"_index": i, **_indexed(x, at_least)} if isinstance(x, dict) else {"_index": i, "value": _indexed(x, at_least)}
                for i, x in enumerate(v)]
    if isinstance(v, dict):
        return {k: _indexed(x, at_least) for k, x in v.items()}
    return v


def state_text(state: JSONContent, letters: bool = False) -> str:
    """The bilinear checkpoints were trained on indented JSON; the letters readout reads one compact
    line with indexed arrays, as decider does."""
    if isinstance(state, str) or not letters:
        return _text(state)
    return json.dumps(_indexed(state), ensure_ascii=False)


def to_questions(req: EvaluateRequest, letters: bool = False) -> list[Question]:
    """With the letters readout the true/false descriptions of a yes/no question are its options
    ("no: ...", "yes: ..."), and JSON is written compactly."""
    out = []
    indent = None if letters else 2
    for qid, q in req.questions.items():
        instr = _text(q.instructions, indent)
        if isinstance(q, BooleanQuestion):
            options = list(BOOLEAN_OPTIONS)
            if q.criteria and letters and set(q.criteria) <= {"true", "false"}:
                options = [o if q.criteria.get(k) in (None, "") else f"{o}: {_text(q.criteria[k], None)}"
                           for o, k in zip(BOOLEAN_OPTIONS, ("false", "true"))]
            elif q.criteria:
                instr += "\n" + "\n".join(f"{k}: {_text(v, indent)}" for k, v in q.criteria.items())
            out.append(Question(qid, q.type, instr, options))
        elif isinstance(q, ChoiceQuestion):
            options = [f"{k}: {_text(v, indent).strip()}" if v not in (None, "") else k for k, v in q.criteria.items()]
            out.append(Question(qid, "choice", instr, options, keys=list(q.criteria)))
        else:
            out.append(Question(qid, "score", instr, [_text(c, indent) for c in q.criteria]))
    return out


def to_answers(questions: list[Question], probs) -> dict[str, dict[str, Any]]:
    """`confidence` is System One's definition, not the top probability: 0 for a uniform spread
    and 1 for certainty (choice), or how tightly a score clusters around its most likely level."""
    answers = {}
    for q, p in zip(questions, probs):
        p = p.tolist()
        n = len(p)
        best = max(range(n), key=p.__getitem__)
        if q.type in ("noul", "boolean"):
            answers[q.qid] = {"type": q.type, "noul" if q.type == "noul" else "probability": round(p[1], 4)}
        elif q.type == "choice":
            keys = q.keys or q.options
            answers[q.qid] = {"type": "choice", "choice": keys[best], "confidence": round((p[best] - 1 / n) / (1 - 1 / n), 4),
                              "probabilities": {k: round(v, 4) for k, v in zip(keys, p)}}
        else:
            spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n  # of a uniform distribution
            answers[q.qid] = {"type": "score", "score": round(sum(i * v for i, v in enumerate(p)), 4),
                              "legend": {str(i): o for i, o in enumerate(q.options)},
                              "probabilities": {str(i): round(v, 4) for i, v in enumerate(p)},
                              "confidence": round(max(0.0, 1 - sum(v * abs(i - best) for i, v in enumerate(p)) / spread), 4)}
    return answers


class Models:
    """The loaded models by name. One forward pass at a time per device: models that share a GPU
    gain nothing from overlapping, and some backends (Apple's MPS) fail if two threads try."""

    def __init__(self, models: dict[str, Agr]) -> None:
        self.models = models
        self.default = next(iter(models))
        per_device: dict[str, threading.Lock] = {}
        self.locks = {name: per_device.setdefault(str(m.device), threading.Lock()) for name, m in models.items()}

    def name(self, requested: str) -> str:
        """An empty name, "agr" or "agr-latest" means the default model; any other unknown name
        is an error."""
        if requested in ("", "agr", "agr-latest"):
            return self.default
        if requested not in self.models:
            raise RequestError(404, f"unknown model {requested!r}; serving {list(self.models)}")
        return requested

    def _answer(self, name: str, req: EvaluateRequest) -> dict[str, Any]:
        model = self.models[name]
        letters = model.readout == "letters"
        questions = to_questions(req, letters)
        try:
            enc = model.encode(state_text(req.state, letters), questions)
        except ValueError as e:
            raise RequestError(422, str(e)) from None
        limit = model.config.get("max_request_tokens", MAX_TOKENS)  # a long-context checkpoint raises it
        if len(enc.ids) > limit:
            raise RequestError(422, f"request is {len(enc.ids)} tokens; the limit is {limit}")
        with self.locks[name]:
            t0 = time.perf_counter()
            probs = model.run(enc)
            ms = (time.perf_counter() - t0) * 1000
        return {"model": name, "answers": to_answers(questions, probs), "truncated": enc.truncated,
                "usage": {"input_tokens": len(enc.ids), "output_tokens": 0}, "latency_ms": round(ms, 2)}

    async def answer(self, req: EvaluateRequest) -> dict[str, Any]:
        return await asyncio.to_thread(self._answer, self.name(req.model), req)


class BodyLimit:
    """Reads the whole request body before the app sees it and refuses it past `limit` bytes. The
    size is counted as the body arrives, so a chunked upload with no Content-Length is capped too."""

    def __init__(self, app, limit: int = MAX_BODY_BYTES) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size, more = [], 0, True
        while more:
            message = await receive()
            if message["type"] != "http.request":  # the client went away
                return
            chunks.append(message.get("body", b""))
            size += len(chunks[-1])
            if size > self.limit:
                return await JSONResponse({"error": "request body too large"}, status_code=413)(scope, receive, send)
            more = message.get("more_body", False)
        replayed = False

        async def replay():
            nonlocal replayed
            if replayed:
                return await receive()  # after the body, only a disconnect can arrive
            replayed = True
            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

        await self.app(scope, replay, send)


def create_app(models: Models, api_key: str | None = None):
    """`api_key`, when set, is required as `Authorization: Bearer <key>` on every route except
    /healthz and the demo pages themselves, which carry no data."""
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse, RedirectResponse

    from .demo.snake_game import DIRS, Game, questions as snake_questions

    app = FastAPI(title="Agr", version="0.1.0")
    pages = Path(__file__).parent / "demo"
    public = {"/healthz", "/demo", "/demo/snake"}

    @app.exception_handler(RequestError)
    async def request_error(_: Request, e: RequestError):
        return JSONResponse({"error": str(e)}, status_code=e.status)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if api_key and request.url.path not in public:
            presented = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(presented.encode(), api_key.encode()):
                return JSONResponse({"error": "missing or wrong key"}, status_code=401)
        return await call_next(request)

    app.add_middleware(BodyLimit)  # added last, so it runs first

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "models": list(models.models)}

    @app.get("/v1/models")
    def list_models():
        return {"data": [{"id": n, "backbone": m.config["backbone"], "default": n == models.default}
                         for n, m in models.models.items()]}

    @app.post("/v1/systemone")
    @app.post("/v1/evaluate")
    async def evaluate(req: EvaluateRequest):
        return await models.answer(req)

    # ------------------------------------------------------------------ demos
    # The game lives on the server between requests, keyed by a session id the page generates;
    # one request plays several moves, and requests for one session run one at a time.

    sessions: dict[str, dict] = {}

    def game_for(key: str, make) -> dict:
        if key not in sessions:
            if len(sessions) > 64:
                sessions.pop(next(iter(sessions)))
            sessions[key] = {**make(), "lock": asyncio.Lock()}
        return sessions[key]

    @app.get("/demo")
    def demo():
        return RedirectResponse("/demo/snake")

    @app.get("/demo/snake", response_class=HTMLResponse)
    def snake_page():
        return HTMLResponse((pages / "snake.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/demo/snake/frames")
    async def snake_frames(request: Request, session: str = Query(min_length=8), n: int = 12, mode: str = "legal",
                           detail: str = "full", model: str = ""):
        name = models.name(model)
        s = game_for(f"snake:{session}:{name}", lambda: {"rng": (r := random.Random()), "game": Game(r), "deaths": 0, "tick": 0})
        async with s["lock"]:
            return {"frames": await play_snake(request, s, name, n, mode, detail)}

    async def play_snake(request: Request, s: dict, name: str, n: int, mode: str, detail: str) -> list[dict]:
        qs = snake_questions(detail)
        frames = []
        for _ in range(max(1, min(48, n))):
            if await request.is_disconnected():  # the page paused or left: stop spending the GPU
                break
            game = s["game"]
            state = game.state(detail)
            res = await models.answer(EvaluateRequest(state=state, questions=qs, model=name))
            a = res["answers"]
            probs = a["move"]["probabilities"]
            safe = {d: a[f"safe_{d}"]["probability"] for d in DIRS}
            truth = {d: game.truth_safe(d) for d in DIRS}
            move, reason = game.choose(probs, safe, mode)
            alive = game.step(move)
            s["tick"] += 1
            s["deaths"] += not alive
            # a copy: the game keeps mutating its body list, and the batch is serialised at the end
            frames.append({"tick": s["tick"], "snake": list(game.snake), "food": game.food, "dir": game.dir,
                           "score": game.score, "deaths": s["deaths"], "move": move, "reason": reason,
                           "probs": probs, "safe": safe, "truth": truth, "alive": alive,
                           "server_ms": res["latency_ms"], "tokens": res["usage"]["input_tokens"], "state": state,
                           "model": name, "backbone": models.models[name].config["backbone"]})
            if not alive:
                s["game"] = Game(s["rng"])
        return frames

    return app


def run(args) -> None:
    import torch
    import uvicorn

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    specs = []  # (name, checkpoint, device)
    if args.models:
        for item in args.models.split(","):
            name, _, rest = item.partition("=")
            ckpt, _, dev = rest.partition("@")
            specs.append((name.strip(), ckpt.strip(), dev.strip() or device))
    elif args.checkpoint:
        specs.append(("agr", args.checkpoint, device))
    if not specs or not all(name and ckpt for name, ckpt, _ in specs):
        raise SystemExit("pass a checkpoint, or --models 'agr=CommandCode/agr@cuda:0,agr-flash=CommandCode/agr-flash@cuda:0'")

    from huggingface_hub.errors import RepositoryNotFoundError

    loaded = {}
    for name, ckpt, dev in specs:
        try:
            # per model: `--models a=repo@cpu` on a GPU machine still wants fp32
            dtype = torch.float32 if dev == "cpu" else torch.bfloat16
            loaded[name] = Agr.from_pretrained(ckpt, device=dev, dtype=dtype)
        except RepositoryNotFoundError:
            raise SystemExit(f"{ckpt}: no such local checkpoint or HuggingFace repo. "
                             "If it is private, run `hf auth login` first.") from None
        print(f"[agr] {name} = {ckpt} on {dev}", flush=True)
    print(f"[agr] http://{args.host}:{args.port}/v1/systemone", flush=True)
    uvicorn.run(create_app(Models(loaded), api_key=args.api_key or None), host=args.host, port=args.port, log_level="info")


def add_args(p) -> None:
    p.add_argument("checkpoint", nargs="?", default="", help="a HuggingFace repo id or a local checkpoint")
    p.add_argument("--models", default="", help="several models on one port: 'agr=CommandCode/agr@cuda:0,agr-flash=CommandCode/agr-flash@cuda:0'")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=os.environ.get("AGR_API_KEY", ""), help="require this key on every route but /healthz and the demo pages")
