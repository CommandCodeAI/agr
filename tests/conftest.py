"""Tiny random checkpoints in the published layout, so the tests need no private weights: a
Llama-shaped one like Agr-flash, and a Gemma 4-shaped one with the sliding-window layers and
per-layer inputs that Agr uses."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer, Gemma4TextConfig, LlamaConfig

from agr.model import Agr, Scorer

TOKENIZER = "HuggingFaceTB/SmolLM2-360M"  # public; only its tokenizer is downloaded


def backbone_config(kind: str, vocab: int):
    if kind == "llama":
        return LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048)
    # per_layer_config={} drops the defaults' overrides for layers a 3-layer model does not have
    return Gemma4TextConfig(vocab_size=vocab, vocab_size_per_layer_input=vocab, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=3, layer_types=["sliding_attention", "sliding_attention", "full_attention"],
                            num_attention_heads=2, num_key_value_heads=1, head_dim=16, hidden_size_per_layer_input=8,
                            sliding_window=8, num_kv_shared_layers=0, per_layer_config={})


@pytest.fixture(scope="session", params=["llama", "gemma4"])
def model(request, tmp_path_factory) -> Agr:
    root = tmp_path_factory.mktemp(request.param)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    AutoModel.from_config(backbone_config(request.param, len(tokenizer))).save_pretrained(root / "backbone")
    tokenizer.save_pretrained(root / "backbone")
    scorer = Scorer(32, 16)
    for p in scorer.parameters():
        torch.nn.init.normal_(p, std=0.5)
    save_file(scorer.state_dict(), str(root / "head.safetensors"))
    (root / "config.json").write_text(json.dumps({"backbone": f"tiny-{request.param}", "head_dim": 16,
                                                  "max_state_tokens": 512, "max_question_tokens": 128}))
    m = Agr.from_pretrained(root, device="cpu", dtype=torch.float32)
    assert (m.window is not None) == (request.param == "gemma4") and m.per_layer_inputs == (request.param == "gemma4")
    return m
