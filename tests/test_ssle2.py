"""Tests for the SSLE-2 Nexus stack: tokenizer, data, model, training step."""

import random

import numpy as np

from nn.module import parameter_count
from ssle2.data import build_sample, make_batches
from ssle2.generate import generate
from ssle2.model import NexusConfig, NexusLM
from ssle2.serialize import load_model, save_model
from ssle2.tokenizer import EOS_ID, SEP_ID, BPETokenizer
from ssle2.trainer import evaluate, masked_cross_entropy

CORPUS = [
    "anvisa proíbe lote de medicamento após denúncia de consumidores",
    "ministério do turismo lança campanha para o litoral nordeste",
    "seleção brasileira vence e avança na copa do mundo",
    "festival de música reúne grandes artistas no fim de semana",
    "governo anuncia novo plano para a saúde pública no país",
    "economia brasileira cresce acima do esperado no trimestre",
    "saiba como esse truque incrível vai mudar a sua rotina hoje",
    "estudo revela benefícios do café para a concentração diária",
] * 8


def _tokenizer():
    tk = BPETokenizer()
    tk.train(CORPUS, vocab_size=300, min_freq=1, themes=["ESPORTE", "SAUDE"])
    return tk


def test_bpe_roundtrip_is_lossless_on_known_words():
    tk = _tokenizer()
    text = "anvisa proíbe lote de medicamento"
    decoded = tk.decode(tk.encode_text(text))
    assert decoded == text


def test_special_and_theme_ids_distinct():
    tk = _tokenizer()
    assert tk.theme_id("ESPORTE") != tk.theme_id("SAUDE")
    assert tk.theme_id("ESPORTE") >= len(tk.merges)  # themes appended last-ish
    assert SEP_ID < tk.size and EOS_ID < tk.size


def test_build_sample_masks_prompt():
    tk = _tokenizer()
    rng = random.Random(0)
    ids, mask = build_sample(tk, "SAUDE", CORPUS[0], rng, max_len=40)
    assert len(ids) == len(mask)
    # SEP marks the prompt/body boundary; everything up to & incl SEP is masked 0.
    sep = ids.index(SEP_ID)
    assert all(m == 0 for m in mask[: sep + 1])
    assert any(m == 1 for m in mask[sep + 1:])


def test_model_forward_shape_and_params():
    tk = _tokenizer()
    cfg = NexusConfig(vocab_size=tk.size, dim=32, heads=4, reasoning_steps=2,
                      memory_slots=8, ffn_mult=2, max_len=40)
    model = NexusLM(cfg, seed=0)
    idx = np.zeros((2, 7), dtype=np.int64)
    logits = model(idx)
    assert logits.shape == (2, 7, tk.size)
    assert parameter_count(model) > 0


def test_training_reduces_loss():
    tk = _tokenizer()
    rng = random.Random(0)
    samples = [build_sample(tk, "SAUDE", t, rng, 40) for t in CORPUS]
    cfg = NexusConfig(vocab_size=tk.size, dim=48, heads=4, reasoning_steps=2,
                      memory_slots=8, ffn_mult=2, max_len=40)
    model = NexusLM(cfg, seed=0)
    batches = list(make_batches(samples, 8, 40, rng, shuffle=False))
    before = evaluate(model, batches)

    from nn.optim import Adam
    opt = Adam(list(model.parameters()), lr=3e-3)
    for _ in range(40):
        for idx, mask in make_batches(samples, 8, 40, rng):
            inp, tgt, wm = idx[:, :-1], idx[:, 1:], mask[:, 1:]
            B, T = inp.shape
            logits = model(inp)
            loss = masked_cross_entropy(
                logits.reshape(B * T, logits.shape[-1]),
                tgt.reshape(B * T).astype(np.int64), wm.reshape(B * T))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss.free_graph()
    after = evaluate(model, batches)
    assert after < before * 0.6, (before, after)


def test_generate_and_serialize_roundtrip(tmp_path):
    tk = _tokenizer()
    cfg = NexusConfig(vocab_size=tk.size, dim=32, heads=4, reasoning_steps=2,
                      memory_slots=8, ffn_mult=2, max_len=40)
    model = NexusLM(cfg, seed=0)
    path = str(tmp_path / "m.nx")
    save_model(path, model, tk)
    model2, tk2 = load_model(path)

    idx = np.array([[tk.theme_id("SAUDE"), SEP_ID]], dtype=np.int64)
    a = model(idx).data
    b = model2(idx).data
    assert np.allclose(a, b, atol=1e-5)

    txt = generate(model2, tk2, "ESPORTE", ["copa"], max_new=12, seed=0)
    assert isinstance(txt, str)
