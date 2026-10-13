import numpy as np
import pytest

from soulsync.emotion import (
    CLIP_SAMPLES,
    LABELS,
    EmotionDetector,
    combine,
    make_windows,
    map_e2v_scores,
    normalize_clip,
    normalize_e2v_label,
)

# The exact label strings shipped in models/emotion2vec/tokens.txt.
E2V_TOKENS = [
    "生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy", "中立/neutral",
    "其他/other", "难过/sad", "吃惊/surprised", "<unk>",
]


class Fixed:
    def __init__(self, probs):
        self.probs = np.array(probs, dtype=float)

    def predict_proba(self, wave):
        return self.probs


def test_bilingual_labels_map_to_our_classes():
    assert normalize_e2v_label("生气/angry") == "angry"
    assert normalize_e2v_label("开心/happy") == "happy"
    assert normalize_e2v_label("中立/neutral") == "neutral"
    assert normalize_e2v_label("难过/sad") == "sad"


def test_fearful_maps_to_anxious():
    assert normalize_e2v_label("恐惧/fearful") == "anxious"


@pytest.mark.parametrize("raw", ["厌恶/disgusted", "吃惊/surprised", "其他/other", "<unk>"])
def test_classes_without_a_counterpart_are_dropped(raw):
    assert normalize_e2v_label(raw) is None


def test_unknown_label_fails_loudly():
    with pytest.raises(KeyError):
        normalize_e2v_label("困惑/confused")


def test_tokens_file_matches_the_mapping():
    from pathlib import Path

    tokens = Path(__file__).parent.parent / "models" / "emotion2vec" / "tokens.txt"
    if not tokens.exists():
        pytest.skip("tokens.txt not present")
    assert tokens.read_text(encoding="utf-8").split() == E2V_TOKENS


def test_scores_are_nonzero_and_renormalised():
    """Regression test for v1, where every Emotion2Vec score was dropped."""
    scores = [0.1, 0.2, 0.4, 0.05, 0.05, 0.05, 0.1, 0.05, 0.0]
    probs = map_e2v_scores(E2V_TOKENS, scores)
    assert probs.sum() == pytest.approx(1.0)
    assert LABELS[int(np.argmax(probs))] == "anxious"
    kept = 0.1 + 0.4 + 0.05 + 0.05 + 0.1
    assert probs[LABELS.index("anxious")] == pytest.approx(0.4 / kept)
    assert probs[LABELS.index("angry")] == pytest.approx(0.1 / kept)


def test_all_mass_on_dropped_classes_gives_uniform():
    probs = map_e2v_scores(["厌恶/disgusted", "吃惊/surprised"], [0.7, 0.3])
    assert np.allclose(probs, 0.2)


def test_combine_weights():
    a, b = np.array([1.0, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0])
    assert np.allclose(combine(a, b, 1.0), a)
    assert np.allclose(combine(a, b, 0.0), b)
    assert np.allclose(combine(a, b, 0.6), [0.6, 0.4, 0, 0, 0])


def test_ensemble_can_change_the_prediction():
    wave = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    w2v = Fixed([0.4, 0.35, 0.1, 0.1, 0.05])
    e2v = Fixed([0.0, 0.9, 0.0, 0.1, 0.0])
    assert EmotionDetector(w2v).detect(wave).label == "happy"
    assert EmotionDetector(w2v, e2v, w2v_weight=0.6).detect(wave).label == "sad"


def test_silence_is_neutral_and_flagged_as_silence():
    result = EmotionDetector(Fixed([0.9, 0.025, 0.025, 0.025, 0.025])).detect(np.zeros(16000, dtype=np.float32))
    assert (result.label, result.source) == ("neutral", "silence")


def test_missing_model_is_reported_not_hidden():
    result = EmotionDetector(None).detect(np.ones(16000, dtype=np.float32))
    assert (result.label, result.source, result.display) == ("unavailable", "unavailable", "Unavailable")


def test_short_clip_is_normalised_then_zero_padded():
    clip = np.random.default_rng(0).normal(3.0, 2.0, 16000).astype(np.float32)
    out = normalize_clip(clip)
    assert out.shape == (CLIP_SAMPLES,)
    assert out[:16000].mean() == pytest.approx(0.0, abs=1e-4)
    assert out[:16000].std() == pytest.approx(1.0, abs=1e-3)
    assert not out[16000:].any()


@pytest.mark.parametrize("seconds,expected", [(1, 1), (3, 1), (4.5, 2), (6, 3), (60, 8)])
def test_window_count(seconds, expected):
    wave = np.random.default_rng(0).normal(0, 0.1, int(seconds * 16000)).astype(np.float32)
    windows = make_windows(wave)
    assert windows.shape == (expected, CLIP_SAMPLES)
