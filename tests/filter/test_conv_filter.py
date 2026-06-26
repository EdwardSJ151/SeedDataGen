"""Tests for filter/phase_conv_filter.py — auto single/multi-turn floor + refusal truncation."""

from SeedDataGen.config import REFUSAL_STRING
from SeedDataGen.filter.phase_conv_filter import ConvFilterConfig, _filter_conversation


def _cfg(**overrides) -> ConvFilterConfig:
    defaults = dict(
        min_messages=4,
        assistant_min_len=10,
        user_levenshtein_threshold=20,
        adjacent_levenshtein_threshold=20,
        batch_size=32,
    )
    defaults.update(overrides)
    return ConvFilterConfig(**defaults)


def _single_turn():
    return [
        {"role": "user", "content": "O que define uma chave fusivel repetidora?"},
        {"role": "assistant", "content": "A CFR e um dispositivo de protecao que reconecta a linha."},
    ]


def _multi_turn(n_pairs: int = 2):
    qs = [
        "O que define uma chave fusivel repetidora?",
        "Quando a protecao seletiva deve atuar nesse contexto?",
        "Como coordenar dispositivos de protecao vizinhos?",
    ]
    ans = [
        "A CFR reconecta a linha apos uma falta transitoria.",
        "A seletiva isola apenas o trecho em falta sem afetar a rede.",
        "A coordenacao exige ajuste temporal entre as curvas dos dispositivos.",
    ]
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": qs[i]})
        msgs.append({"role": "assistant", "content": ans[i]})
    return msgs


def test_single_turn_kept_by_default():
    # Auto-detected single-turn (1 user turn) → floor 2, kept with default cfg.
    kept = _filter_conversation(_single_turn(), _cfg())
    assert kept is not None and len(kept) == 2


def test_multi_turn_two_pairs_kept():
    kept = _filter_conversation(_multi_turn(2), _cfg())
    assert kept is not None and len(kept) == 4


def test_multi_turn_three_pairs_kept_floor_stays_four():
    kept = _filter_conversation(_multi_turn(3), _cfg())
    assert kept is not None and len(kept) == 6


def test_single_turn_refusal_dropped():
    msgs = [
        {"role": "user", "content": "Pergunta fora de escopo?"},
        {"role": "assistant", "content": REFUSAL_STRING},
    ]
    # truncation removes the refusing pair → empty → dropped.
    assert _filter_conversation(msgs, _cfg()) is None


def test_multi_turn_collapsed_by_early_refusal_dropped():
    # 2-pair convo whose 2nd answer is a refusal: input has 2 user turns (multi-turn
    # intent), truncates to the seed pair (2 msgs) → below the multi-turn floor (4) → dropped.
    msgs = _multi_turn(2)
    msgs[3] = {"role": "assistant", "content": REFUSAL_STRING}
    assert _filter_conversation(msgs, _cfg()) is None
