"""Score any System One endpoint on a labelled file, and compare two scored runs question by question.

    agr eval http://localhost:8000 labelled.jsonl --out agr.jsonl
    agr compare agr.jsonl other.jsonl

Each line of a labelled file is a System One request whose questions also carry a `label`:

    choice  one of the option names
    noul    true or false (the strings "true" and "false" work too)
    score   0 for the first level, 1 for the next, and so on
`compare` pairs questions by the request's content, not its line number, so two runs line up
even when one file was shuffled or filtered.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


def ask(url: str, body: dict, key: str) -> tuple[dict, float | None, float]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out["answers"], out.get("latency_ms"), (time.perf_counter() - t0) * 1000


YES_NO = {True: 1, False: 0, 1: 1, 0: 0, "true": 1, "false": 0}


def yes_no(label) -> int:
    key = label.strip().lower() if isinstance(label, str) else label
    if key not in YES_NO:
        raise ValueError(f"a yes/no label must be true or false, got {label!r}")
    return YES_NO[key]


def read_lines(path: str) -> list[dict]:
    # split on "\n" only: str.splitlines() also breaks on U+2028 and friends, which JSON allows raw
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def request_key(state, questions: dict) -> str:
    """Identifies a request by its content, so two runs only pair up on the same request."""
    return hashlib.sha256(json.dumps({"state": state, "questions": questions}, sort_keys=True).encode()).hexdigest()[:16]


def scored(q: dict, answer: dict) -> tuple[list[float], int]:
    """The answer as probabilities in the question's own option order, and the label's index."""
    if q["type"] in ("noul", "boolean"):
        p = float(answer.get("noul", answer.get("probability")))
        return [1 - p, p], yes_no(q["label"])
    if q["type"] == "choice":
        names = list(q["criteria"])
        return [float(answer["probabilities"].get(n, 0.0)) for n in names], names.index(q["label"])
    return [float(answer["probabilities"].get(str(i), 0.0)) for i in range(len(q["criteria"]))], int(q["label"])


def evaluate(args) -> None:
    lines = read_lines(args.file)[: args.limit or None]
    url = args.url.rstrip("/") + f"/v1/{args.route}"
    rows, server, http = [], [], []
    for rec in lines:
        questions = {}
        for qid, q in rec["questions"].items():
            q = {k: v for k, v in q.items() if k != "label"}
            if args.route == "evaluate" and q["type"] == "noul":  # Vercel AI Gateway's spelling
                q["type"] = "boolean"
            questions[qid] = q
        key = request_key(rec["state"], {qid: {k: v for k, v in q.items() if k != "label"} for qid, q in rec["questions"].items()})
        answers, server_ms, http_ms = ask(url, {"model": args.model, "state": rec["state"], "questions": questions}, args.key)
        http.append(http_ms)
        if server_ms is not None:
            server.append(server_ms)
        for qid, q in rec["questions"].items():
            if "label" in q:
                probs, truth = scored(q, answers[qid])
                pred = max(range(len(probs)), key=probs.__getitem__)
                rows.append({"record": key, "qid": qid, "type": q["type"], "label": truth, "pred": pred,
                             "correct": pred == truth, "probs": probs})
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    for name, group in [("all", rows), *sorted(by_type.items())]:
        print(f"{name:8} n={len(group):5}  accuracy {mean(r['correct'] for r in group):.3f}  "
              f"brier {mean(brier(r) for r in group):.3f}  ece {ece(group):.3f}")
    for name, ms in [("server", server), ("http", http)]:
        if ms:
            p90 = sorted(ms)[math.ceil(0.9 * len(ms)) - 1]  # nearest rank: never below the median
            print(f"{name} latency ms: p50 {statistics.median(ms):.1f}  p90 {p90:.1f}")


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def brier(r: dict) -> float:
    return sum((p - (i == r["label"])) ** 2 for i, p in enumerate(r["probs"]))


def ece(rows: list[dict], bins: int = 10) -> float:
    """Expected calibration error of the top answer's probability."""
    buckets = defaultdict(list)
    for r in rows:
        top = max(r["probs"])
        buckets[min(int(top * bins), bins - 1)].append((top, r["correct"]))
    return sum(len(b) / len(rows) * abs(mean(t for t, _ in b) - mean(c for _, c in b)) for b in buckets.values())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for an accuracy of k out of n."""
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided p-value that two models are equally accurate, from the questions only one
    of them got right."""
    n, k = only_a + only_b, min(only_a, only_b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def compare(args) -> None:
    runs = [{(r["record"], r["qid"]): r["correct"] for r in read_lines(f)} for f in (args.a, args.b)]
    shared = runs[0].keys() & runs[1].keys()
    n = len(shared)
    if not n:
        raise SystemExit("the two runs share no questions: were they scored on the same file?")
    for name, run in zip((args.a, args.b), runs):
        k = sum(run[q] for q in shared)
        lo, hi = wilson(k, n)
        print(f"{name}: {k}/{n} = {k / n:.3f}  (95% interval {lo:.3f} to {hi:.3f})")
    only_a = sum(runs[0][q] and not runs[1][q] for q in shared)
    only_b = sum(runs[1][q] and not runs[0][q] for q in shared)
    print(f"right only in the first: {only_a}, only in the second: {only_b}, paired p = {mcnemar(only_a, only_b):.4f}")


def add_args(eval_parser, compare_parser) -> None:
    eval_parser.add_argument("url", help="the server, e.g. http://localhost:8000")
    eval_parser.add_argument("file", help="labelled requests, one per line")
    eval_parser.add_argument("--out", default="", help="write one row per question here, for `agr compare`")
    eval_parser.add_argument("--model", default="")
    eval_parser.add_argument("--key", default="local", help="bearer key, if the server wants one")
    eval_parser.add_argument("--route", default="systemone", choices=["systemone", "evaluate"])
    eval_parser.add_argument("--limit", type=int, default=0, help="only the first N requests")
    compare_parser.add_argument("a")
    compare_parser.add_argument("b")
