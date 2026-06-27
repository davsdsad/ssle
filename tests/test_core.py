"""Unit tests for the SSLE-1 core engine."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from core.dataset import parse_sds2
from core.engine import EngineConfig, SSLEEngine
from core.matrix import TransitionMatrix
from core.sampler import softmax
from core.theme import ThemeProfiles
from core.tokenizer import BOS_ID, EOS_ID, Tokenizer, normalize
from core.trainer import Trainer

SAMPLE_SDS2 = """
[SAMPLE]
theme=FORTNITE
context=MIRA|RANKED|DICAS
target=COMO MELHORAR SUA MIRA NO FORTNITE
weight=1.5
[/SAMPLE]

[SAMPLE]
theme=ACADEMIA
context=TREINO|DIETA
target=5 DICAS PARA GANHAR MASSA MUSCULAR
weight=1.0
[/SAMPLE]
"""


def test_normalize_removes_accents_and_punct():
    assert normalize("Olá, Mundo!") == "OLA MUNDO"
    assert normalize("  café   com   leite  ") == "CAFE COM LEITE"


def test_tokenizer_roundtrip():
    tok = Tokenizer(max_vocab=100)
    tok.count(["COMO MELHORAR SUA MIRA", "MELHORAR A MIRA"])
    tok.build()
    ids = tok.encode("MELHORAR MIRA")
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    decoded = tok.decode(ids)
    assert "MELHORAR" in decoded and "MIRA" in decoded


def test_parse_sds2():
    samples = parse_sds2(SAMPLE_SDS2)
    assert len(samples) == 2
    assert samples[0].theme == "FORTNITE"
    assert samples[0].context == ["MIRA", "RANKED", "DICAS"]
    assert samples[0].weight == 1.5
    assert "MIRA" in samples[0].target


def test_softmax_sums_to_one():
    p = softmax(np.array([1.0, 2.0, 3.0]), temperature=0.8)
    assert pytest.approx(float(p.sum()), abs=1e-5) == 1.0
    assert np.all(p >= 0)


def test_matrix_backoff():
    m = TransitionMatrix(n_order=3, smoothing=0.1)
    m.update_counts([5, 6, 7, 8], weight=1.0)
    logits = m.get_logits([5, 6], vocab_size=20)
    assert logits.shape == (20,)
    # An unseen long context should back off, not crash.
    logits2 = m.get_logits([99, 98], vocab_size=20)
    assert logits2.shape == (20,)


def test_theme_profile_prior_is_themed():
    tp = ThemeProfiles()
    tp.observe("FORTNITE", [5, 6, 7], weight=1.0)
    tp.observe("CULINARIA", [8, 9, 10], weight=1.0)
    prior = tp.prior("FORTNITE", vocab_size=20)
    # FORTNITE tokens should score higher than CULINARIA-only tokens.
    assert prior[5] > prior[8]


def test_end_to_end_train_generate_save_load():
    samples = parse_sds2(SAMPLE_SDS2)
    config = EngineConfig(n_order=2, embedding_dim=16, vocab_size=200)
    engine = SSLEEngine(config, seed=1)
    engine.tokenizer.count([s.target for s in samples])
    engine.tokenizer.build()
    engine.sync_embeddings(seed=1)
    stats = Trainer(engine, seed=1).train(samples, epochs=2, log=lambda m: None)
    assert stats["final_loss"] is not None

    out = engine.generate(theme="FORTNITE", context="MIRA", max_tokens=10, seed=1)
    assert isinstance(out, str) and len(out) > 0

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.snm")
        engine.save(path)
        assert os.path.getsize(path) > 0
        loaded = SSLEEngine.load(path)
        assert loaded.tokenizer.vocab_size == engine.tokenizer.vocab_size
        out2 = loaded.generate(theme="FORTNITE", context="MIRA", max_tokens=10, seed=1)
        assert isinstance(out2, str)
