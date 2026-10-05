"""Load an Agr checkpoint and score the choices of many typed questions about one document, all in
a single forward pass of the transformer.

Two readouts. `bilinear` (the default) wraps the request in learned marker tokens and scores each
choice with a small query/key head. `letters` keeps an instruction-tuned backbone's own chat format:
each question lists its options under single-token codes (A, B, ..., then AA, AB, ...) and the
answer is the backbone's own next-token distribution over those codes, read where its reply would
begin. Either way every question is its own branch after the shared document."""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from .records import Question

# Tokens the base vocabulary does not have. Their embeddings are learned and stored with the scorer.
MARKERS = ("<|agr:doc|>", "<|agr:ask|>", "<|agr:choice|>", "<|agr:/choice|>", "<|agr:answer|>")
DOC, ASK, CHOICE, CHOICE_END, ANSWER = range(len(MARKERS))


@dataclass
class Layout:
    """A request as one token sequence: the document, then a block per question. A block sits at
    the positions right after the document and sees the document and itself, never another block,
    so an answer is the same whichever other questions come with it."""

    ids: list[int]
    positions: list[int]
    doc_len: int
    blocks: list[tuple[int, int]] = field(default_factory=list)  # [start, end) of each question
    choices: list[list[tuple[int, int]]] = field(default_factory=list)  # [start, end) of each choice
    codes: list[list[int]] = field(default_factory=list)  # letters readout: each option's code token
    mirror: list[bool] = field(default_factory=list)  # the block before's question, options reversed
    truncated: bool = False  # the document or a question was cut to fit

    def add(self, block: list[int], choices: list[tuple[int, int]], codes: list[int] | None = None,
            mirror: bool = False) -> None:
        start = len(self.ids)
        self.ids += block
        self.positions += range(self.doc_len, self.doc_len + len(block))
        self.blocks.append((start, len(self.ids)))
        self.choices.append([(start + a, start + b) for a, b in choices])
        self.codes.append(codes or [])
        self.mirror.append(mirror)

    def visibility(self, device: torch.device) -> torch.Tensor:
        n, d = len(self.ids), self.doc_len

        def lower(k: int) -> torch.Tensor:
            return torch.ones(k, k, dtype=torch.bool, device=device).tril()

        see = torch.zeros(n, n, dtype=torch.bool, device=device)
        see[:d, :d] = lower(d)
        for a, b in self.blocks:
            see[a:b, :d] = True
            see[a:b, a:b] = lower(b - a)
        return see


class Scorer(nn.Module):
    """The marker embeddings, and a bilinear match between the hidden vector at a question's last
    token and the mean of each choice's tokens."""

    def __init__(self, hidden: int, dim: int) -> None:
        super().__init__()
        self.markers = nn.Parameter(torch.zeros(len(MARKERS), hidden))
        self.query = nn.Linear(hidden, dim, bias=False)
        self.key = nn.Linear(hidden, dim, bias=False)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, h: torch.Tensor, layout: Layout) -> list[torch.Tensor]:
        out = []
        for (_, end), spans in zip(layout.blocks, layout.choices):
            asked = self.query(h[end - 1])
            pooled = torch.stack([h[a:b].mean(0) for a, b in spans])
            out.append((self.key(pooled) @ asked * self.scale).softmax(-1))
        return out


SENTINEL = "AGR"
# The letters prompt is decider's chat layout (github.com/Mapika/decider, Apache-2.0): the document
# opens the user turn, each question lists its options as "(A) text", and the reply begins "Answer: (".
LETTERS_CONTEXT = "Context:\n"
LETTERS_ASK = "\n\nQuestion: {q}\nOptions:"
LETTERS_ANSWER = "Answer: ("
NARROW = 10  # up to this many options are written out as one string; more get one code token each
THINKING = (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"), ("<|channel>thought\n", "<channel|>"))


def option_codes(tokenizer, limit: int = 255) -> list[tuple[str, int]]:
    """Codes that are one token each in this vocabulary: A to Z, then two-letter pairs."""
    import itertools
    import string

    out = []
    for size in (1, 2):
        for code in map("".join, itertools.product(string.ascii_uppercase, repeat=size)):
            ids = tokenizer.encode(code, add_special_tokens=False)
            if len(ids) == 1:
                out.append((code, ids[0]))
            if len(out) == limit:
                return out
    return out


class Agr(nn.Module):
    def __init__(self, backbone: nn.Module, scorer: Scorer | None, tokenizer, config: dict) -> None:
        super().__init__()
        self.backbone = backbone
        self.scorer = scorer
        self.tokenizer = tokenizer
        self.config = config
        self.readout = config.get("readout", "bilinear")
        self.vocab = backbone.get_input_embeddings().num_embeddings
        cfg = backbone.config
        kinds = set(getattr(cfg, "layer_types", None) or ())
        self.window = cfg.sliding_window if "sliding_attention" in kinds else None
        self.per_layer_inputs = hasattr(backbone, "get_per_layer_inputs") and bool(getattr(cfg, "hidden_size_per_layer_input", 0))
        if self.readout == "letters":
            # the backbone's own output layer (tied to its input embeddings), chat format and option codes
            self.softcap = getattr(cfg, "final_logit_softcapping", None)
            self.codes = option_codes(tokenizer)
            text = tokenizer.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            before, after = text.split(SENTINEL)
            for opened, closed in THINKING:  # a thinking block the template leaves open is closed here
                if after.endswith(opened):
                    after += closed
            self.template = (tokenizer.encode(before, add_special_tokens=False),
                             tokenizer.encode(after, add_special_tokens=False) + self.tokenize(LETTERS_ANSWER))
            self.open = self.tokenize("\n(")
        else:
            tokenizer.add_tokens(list(MARKERS), special_tokens=True)
            self.marker_ids = tokenizer.convert_tokens_to_ids(list(MARKERS))

    @classmethod
    def from_pretrained(cls, repo: str | Path, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> "Agr":
        """A local checkpoint folder or a HuggingFace repo id: `config.json`, the transformer in
        `backbone/`, and (for the bilinear readout) `head.safetensors`."""
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        root = Path(repo) if Path(repo, "config.json").is_file() else Path(snapshot_download(str(repo)))
        config = json.loads((root / "config.json").read_text())
        # SDPA is the attention kernel that applies a custom mask; the others would ignore it.
        backbone = AutoModel.from_pretrained(root / "backbone", dtype=dtype, attn_implementation="sdpa")
        backbone = getattr(backbone, "language_model", backbone)
        tokenizer = AutoTokenizer.from_pretrained(root / "backbone")
        if config.get("readout", "bilinear") == "letters":
            text_cfg = getattr(backbone.config, "text_config", backbone.config)
            if not getattr(text_cfg, "tie_word_embeddings", True) and not getattr(backbone.config, "tie_word_embeddings", True):
                raise ValueError("the letters readout needs an output layer tied to the input embeddings")
            return cls(backbone.requires_grad_(False), None, tokenizer, config).to(device).eval()
        scorer = Scorer(backbone.config.hidden_size, config["head_dim"])
        scorer.load_state_dict(load_file(str(root / "head.safetensors")))
        return cls(backbone.requires_grad_(False), scorer.float(), tokenizer, config).to(device).eval()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def tokenize(self, text: str) -> list[int]:
        # Text in a request stays text: a marker or control token spelled out in it is not one.
        return self.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)

    def encode(self, document: str, questions: Sequence[Question]) -> Layout:
        if self.readout == "letters":
            return self._encode_letters(document, questions)
        bos = self.tokenizer.bos_token_id
        start = ([] if bos is None else [bos]) + [self.marker_ids[DOC]]
        text = self.tokenize(document)
        keep = self.config["max_state_tokens"] - len(start)
        ids = start + text[:keep]
        layout = Layout(ids, list(range(len(ids))), len(ids), truncated=len(text) > keep)
        for q in questions:
            block, spans, cut = self._block(q)
            layout.add(block, spans)
            layout.truncated |= cut
        return layout

    def _block(self, q: Question) -> tuple[list[int], list[tuple[int, int]], bool]:
        """One question's tokens, where its choices sit in them, and whether its text was cut."""
        m = self.marker_ids
        choices = [[m[CHOICE], *self.tokenize(c), m[CHOICE_END]] for c in q.options]
        budget = self.config["max_question_tokens"] - 2 - sum(map(len, choices))
        if budget < 1:
            raise ValueError(f"{q.qid}: the options are too long")
        prompt = self.tokenize(q.instructions)
        block, spans = [m[ASK], *prompt[:budget]], []
        for c in choices:
            spans.append((len(block), len(block) + len(c)))
            block += c
        return block + [m[ANSWER]], spans, len(prompt) > budget

    def _encode_letters(self, document: str, questions: Sequence[Question]) -> Layout:
        """The chat turn opens once, with the document; each question continues it in its own branch,
        lists its options under codes, and closes the turn where the backbone's reply would begin.
        With `order_averaging`, a choice or yes/no question gets a second branch with its options in
        reverse order, and its answer is the mean of the two readings."""
        before, answer = self.template
        text = self.tokenize(LETTERS_CONTEXT + document)
        keep = self.config["max_state_tokens"] - len(before)
        ids = before + text[:keep]
        layout = Layout(ids, list(range(len(ids))), len(ids), truncated=len(text) > keep)
        for q in questions:
            if len(q.options) > len(self.codes):
                raise ValueError(f"{q.qid}: at most {len(self.codes)} options")
            options = [f"{i}: {o}" for i, o in enumerate(q.options)] if q.type == "score" else list(q.options)
            orders = [options, options[::-1]] if self.config.get("order_averaging") and q.type != "score" else [options]
            for k, listed in enumerate(orders):
                block, cut = self._question_letters(q, listed, answer)
                layout.add(block, [], [t for _, t in self.codes[:len(listed)]], mirror=k > 0)
                layout.truncated |= cut
        return layout

    def _question_letters(self, q: Question, options: list[str], answer: list[int]) -> tuple[list[int], bool]:
        if len(options) <= NARROW:
            listed = self.tokenize("".join(f"\n({c}) {o}" for (c, _), o in zip(self.codes, options)))
        else:  # "\n(" + code + ") text", built from ids so that every code stays one token
            listed = [t for (_, c), o in zip(self.codes, options) for t in self.open + [c] + self.tokenize(f") {o}")]
        limit = self.config["max_question_tokens"] - len(listed) - len(answer)
        whole = self.tokenize(LETTERS_ASK.format(q=q.instructions))
        if len(whole) <= limit:
            return whole + listed + answer, False
        ask, rest = LETTERS_ASK.split("{q}")
        head, tail = self.tokenize(ask), self.tokenize(rest)
        budget = limit - len(head) - len(tail)
        if budget < 1:
            raise ValueError(f"{q.qid}: the options are too long")
        return head + self.tokenize(q.instructions)[:budget] + tail + listed + answer, True

    def _attention(self, layout: Layout, dtype: torch.dtype) -> torch.Tensor | dict[str, torch.Tensor]:
        see = layout.visibility(self.device)

        def additive(allowed: torch.Tensor) -> torch.Tensor:
            return torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)[None, None]

        if self.window is None:
            return additive(see)
        # Sliding-window layers also drop keys more than a window behind, by position.
        pos = torch.tensor(layout.positions, device=self.device)
        near = (pos[:, None] - pos[None, :]) < self.window
        return {"full_attention": additive(see), "sliding_attention": additive(see & near)}

    def _inputs(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.readout == "letters":  # no marker tokens: the backbone embeds its own vocabulary
            return {"input_ids": ids}
        slot = torch.full_like(ids, -1)
        for i, m in enumerate(self.marker_ids):
            slot[ids == m] = i
        marked = slot >= 0
        known = ids.clamp_max(self.vocab - 1)  # marker ids lie past the base vocabulary
        x = self.backbone.get_input_embeddings()(known)
        x[marked] = self.scorer.markers.to(x.dtype)[slot[marked]]
        inputs = {"inputs_embeds": x}
        if self.per_layer_inputs:
            inputs["per_layer_inputs"] = self.backbone.get_per_layer_inputs(known, x).masked_fill(marked[..., None, None], 0.0)
        return inputs

    def probs(self, document: str, questions: Sequence[Question]) -> list[torch.Tensor]:
        """One probability vector per question, in the order the questions were given."""
        return self.run(self.encode(document, questions))

    @torch.inference_mode()
    def run(self, layout: Layout) -> list[torch.Tensor]:
        ids = torch.tensor([layout.ids], device=self.device)
        dtype = next(self.backbone.parameters()).dtype
        out = self.backbone(**self._inputs(ids), position_ids=torch.tensor([layout.positions], device=self.device),
                            attention_mask=self._attention(layout, dtype), use_cache=False)
        h = out.last_hidden_state[0].float()
        exact = torch.autocast("cuda", enabled=False) if self.device.type == "cuda" else contextlib.nullcontext()
        with exact:
            if self.readout == "letters":
                out = []
                for (_, end), codes, mirror in zip(layout.blocks, layout.codes, layout.mirror):
                    p = self._letters(h[end - 1], codes).cpu()
                    if mirror:
                        out[-1] = (out[-1] + p.flip(0)) / 2
                    else:
                        out.append(p)
                return out
            return [p.cpu() for p in self.scorer(h, layout)]

    def _letters(self, last: torch.Tensor, codes: list[int]) -> torch.Tensor:
        """The backbone's next-token logits over the option codes, with its own final softcap, at the
        checkpoint's temperature: one value, or T(n) = max(min, a + b ln n) for n options."""
        logits = self.backbone.get_input_embeddings().weight[codes].float() @ last
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        rule = self.config.get("temperature_by_options")
        t = max(rule.get("min", 0.05), rule["a"] + rule["b"] * math.log(max(len(codes), 2))) if rule else self.config.get("temperature", 1.0)
        return (logits / t).softmax(-1)
