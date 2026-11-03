"""
Test cho vi_text (chuẩn hoá tiếng Việt) và text_classifier (TF-IDF + LR).
Quan trọng nhất: inference NumPy thuần trong backend phải cho ra ĐÚNG xác suất
như sklearn lúc train - lệch là model chạy production khác model đã đánh giá.
"""
import numpy as np
import pytest

from emotion_mlp import EMOTIONS
from vi_text import normalize, strip_accents
from text_classifier import CHANNELS, TfidfEmotionModel, _channel_text, train
from train_tfidf import seed_pairs

N = len(EMOTIONS)


@pytest.mark.parametrize("raw, expected", [
    ("Hôm nay VUIIII quá!!!", "hôm nay vui quá"),
    ("ko dc vui lắm", "không được vui lắm"),
    ("mệt wá r", "mệt quá rồi"),
    ("hòa bình, thúy", "hoà bình thuý"),     # dấu thanh kiểu cũ -> kiểu mới
    ("hoà bình, thuý", "hoà bình thuý"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_handles_nfd_input():
    import unicodedata
    nfd = unicodedata.normalize("NFD", "buồn quá")
    assert normalize(nfd) == "buồn quá"


def test_strip_accents():
    assert strip_accents("đường về nhà buồn quá") == "duong ve nha buon qua"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pairs = seed_pairs()
    texts = [t for t, _ in pairs]
    labels = [l for _, l in pairs]
    path = str(tmp_path_factory.mktemp("tfidf") / "model.npz")
    clf, vecs = train(texts, labels, np.ones(len(texts)), N, path)
    return clf, vecs, TfidfEmotionModel.load(path)


SAMPLES = [
    "hôm nay tôi rất vui",
    "hom nay buon qua khong muon lam gi",     # không dấu
    "deadline dí sát nút, stress muốn xỉu",
    "zzz qwerty",                              # toàn OOV
    "",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_numpy_inference_matches_sklearn(trained, text):
    from scipy.sparse import hstack
    clf, vecs, model = trained
    X = hstack([vecs[ch].transform([_channel_text(ch, text)]) for ch in CHANNELS]).tocsr()
    expected = np.zeros(N)
    expected[clf.classes_] = clf.predict_proba(X)[0]
    np.testing.assert_allclose(model.predict_proba(text), expected, atol=1e-9)


def test_unaccented_input_matches_accented_prediction(trained):
    _, _, model = trained
    for text in ["hôm nay tôi rất vui", "buồn lắm không muốn làm gì", "tức giận điên người luôn"]:
        assert model.predict_proba(strip_accents(text)).argmax() == model.predict_proba(text).argmax()


def _bare_engine(path):
    import emotion_mlp
    eng = emotion_mlp.EmotionEngine.__new__(emotion_mlp.EmotionEngine)
    eng.tfidf_path, eng.tfidf = str(path), None
    return eng


def test_corrupt_model_file_does_not_crash_engine(tmp_path):
    bad = tmp_path / "tfidf_model.npz"
    bad.write_bytes(b"PK\x03\x04 truncated")
    eng = _bare_engine(bad)
    eng._load_tfidf()
    assert eng.tfidf is None


def test_model_with_wrong_class_count_is_rejected(trained, tmp_path):
    _, _, model = trained
    np.savez(tmp_path / "m.npz", x=0)  # chỉ cần path tồn tại, load() được patch bên dưới
    eng = _bare_engine(tmp_path / "m.npz")
    model.n_classes = N + 1
    orig = TfidfEmotionModel.load
    try:
        TfidfEmotionModel.load = classmethod(lambda cls, p: model)
        eng._load_tfidf()
    finally:
        TfidfEmotionModel.load = orig
        model.n_classes = N
    assert eng.tfidf is None


def test_pseudo_labels_only_from_before_cutoff(tmp_path):
    import json
    from train_tfidf import load_feedback, PSEUDO_CUTOFF
    log = tmp_path / "log.jsonl"
    rows = [
        {"time": "2026-06-01T10:00:00", "type": "predict", "text": "cũ", "result": "happy"},
        {"time": PSEUDO_CUTOFF + "T10:00:00", "type": "predict", "text": "mới", "result": "sad"},
        {"time": PSEUDO_CUTOFF + "T11:00:00", "type": "feedback", "text": "dạy", "correct": "angry"},
    ]
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    labeled, pseudo = load_feedback(str(log))
    assert labeled == {"dạy": EMOTIONS.index("angry")}
    assert pseudo == {"cũ": EMOTIONS.index("happy")}


def test_retrain_debounce_is_capped_by_max_wait(monkeypatch):
    import emotion_mlp
    delays = []

    class FakeTimer:
        def __init__(self, delay, fn): delays.append(delay); self.daemon = False
        def start(self): pass
        def cancel(self): pass

    monkeypatch.setattr(emotion_mlp.threading, "Timer", FakeTimer)
    clock = [1000.0]
    monkeypatch.setattr(emotion_mlp.time, "time", lambda: clock[0])
    r = emotion_mlp._TfidfRetrainer(lambda: None)
    r.schedule()
    clock[0] += emotion_mlp.TFIDF_RETRAIN_MAX_WAIT - 10   # feedback dồn dập suốt ~10 phút
    r.schedule()
    assert delays[0] == emotion_mlp.TFIDF_RETRAIN_DELAY
    assert delays[1] == 10
