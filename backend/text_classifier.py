"""
TF-IDF n-gram + Logistic Regression cho cảm xúc tiếng Việt (v5.0).

4 kênh đặc trưng, mỗi kênh TF-IDF (sublinear tf, chuẩn hoá L2) riêng rồi nối lại:
  word        : n-gram 1-2 âm tiết trên câu đã chuẩn hoá (vi_text.normalize)
  char        : n-gram ký tự 2-5 trong phạm vi từ (char_wb) - chịu được lỗi chính tả
  noacc       : như char nhưng trên bản bỏ dấu  -> nhận ra câu gõ không dấu
  noacc_word  : như word nhưng trên bản bỏ dấu

Train dùng scikit-learn (train(), gọi từ train_tfidf.py trong tiến trình riêng),
còn lúc chạy chỉ cần NumPy (TfidfEmotionModel): server ít RAM nên backend không
import sklearn/scipy (~80 MB). Kết quả khớp predict_proba của sklearn (xem test).
"""
import os
import re
from collections import Counter

import numpy as np

from vi_text import normalize, strip_accents

MODEL_FILE = "tfidf_model.npz"
# Tăng khi đổi vi_text.normalize / TEENCODE / analyzer: model cũ trên server bị
# bỏ qua và train lại, thay vì âm thầm dùng vocab không còn khớp tiền xử lý.
FEATURE_VERSION = 1
CHANNELS = ("word", "char", "noacc", "noacc_word")
# Giá trị chọn bằng 5-fold CV trên feedback có nhãn (xem BaoCao v5.0)
LR_C = 10.0
NOACC_AUG_WEIGHT = 0.5   # trọng số bản sao bỏ dấu của mỗi câu train
PSEUDO_WEIGHT = 0.3      # trọng số câu chỉ có nhãn do model cũ đoán (type=predict)

_WS_RE = re.compile(r"\s+")


def _channel_text(name, text):
    t = normalize(text)
    return strip_accents(t) if name.startswith("noacc") else t


def _word_ngrams(doc):
    tokens = doc.split()
    return tokens + [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]


def _char_wb_ngrams(doc, min_n=2, max_n=5):
    # Tái hiện đúng TfidfVectorizer(analyzer="char_wb") của sklearn
    grams = []
    for w in _WS_RE.sub(" ", doc).split():
        w = " " + w + " "
        w_len = len(w)
        for n in range(min_n, max_n + 1):
            offset = 0
            grams.append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                grams.append(w[offset:offset + n])
            if offset == 0:
                break
    return grams


def _analyze(name, text):
    doc = _channel_text(name, text)
    return _word_ngrams(doc) if name.endswith("word") else _char_wb_ngrams(doc)


class TfidfEmotionModel:
    """Inference NumPy thuần từ file .npz do train() xuất ra."""

    def __init__(self, vocabs, idfs, coef, intercept, classes, n_classes, feature_version=FEATURE_VERSION):
        self.feature_version = feature_version
        self.vocabs = vocabs          # {channel: {term: col}}
        self.idfs = idfs              # {channel: np.ndarray}
        self.coef = coef              # (n_model_classes, n_features)
        self.intercept = intercept
        self.classes = classes        # nhãn (int) ứng với mỗi hàng coef
        self.n_classes = n_classes
        offsets, off = {}, 0
        for ch in CHANNELS:
            offsets[ch] = off
            off += len(idfs[ch])
        self.offsets = offsets

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            vocabs, idfs = {}, {}
            for ch in CHANNELS:
                terms = str(d[f"{ch}_terms"]).split("\n") if len(d[f"{ch}_idf"]) else []
                vocabs[ch] = {t: i for i, t in enumerate(terms)}
                idfs[ch] = d[f"{ch}_idf"]
            version = int(d["feature_version"]) if "feature_version" in d.files else 0
            return cls(vocabs, idfs, d["coef"], d["intercept"], d["classes"], int(d["n_classes"]), version)

    def _features(self, text):
        cols, vals = [], []
        for ch in CHANNELS:
            vocab, idf = self.vocabs[ch], self.idfs[ch]
            counts = Counter(g for g in _analyze(ch, text) if g in vocab)
            if not counts:
                continue
            idx = np.fromiter((vocab[g] for g in counts), dtype=np.int64, count=len(counts))
            tf = 1.0 + np.log(np.fromiter(counts.values(), dtype=np.float64, count=len(counts)))
            v = tf * idf[idx]
            v /= np.linalg.norm(v)
            cols.append(idx + self.offsets[ch])
            vals.append(v)
        if not cols:
            return np.empty(0, dtype=np.int64), np.empty(0)
        return np.concatenate(cols), np.concatenate(vals)

    def predict_proba(self, text):
        """Xác suất (n_classes,) theo thứ tự EMOTIONS."""
        cols, vals = self._features(text)
        logits = self.coef[:, cols] @ vals + self.intercept
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        full = np.zeros(self.n_classes)
        full[self.classes] = p
        return full


def train(texts, labels, weights, n_classes, out_path):
    """Train bằng sklearn rồi xuất .npz (ghi file tạm rồi rename -> không bao
    giờ để backend đọc phải file ghi dở)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack

    # Thêm bản sao bỏ dấu: dữ liệu train toàn câu có dấu nhưng người dùng
    # hay gõ không dấu -> kênh word/char có dấu cũng thấy dạng không dấu.
    texts = list(texts) + [strip_accents(normalize(t)) for t in texts]
    labels = list(labels) * 2
    weights = np.concatenate([weights, np.asarray(weights) * NOACC_AUG_WEIGHT])

    blocks, arrays, vecs = [], {}, {}
    for ch in CHANNELS:
        docs = [_channel_text(ch, t) for t in texts]
        if ch.endswith("word"):
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+", lowercase=False)
        else:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, lowercase=False)
        blocks.append(vec.fit_transform(docs))
        vecs[ch] = vec
        terms = vec.get_feature_names_out()
        arrays[f"{ch}_terms"] = np.array("\n".join(terms))
        arrays[f"{ch}_idf"] = vec.idf_
    X = hstack(blocks).tocsr()

    clf = LogisticRegression(C=LR_C, max_iter=3000, class_weight="balanced")
    clf.fit(X, labels, sample_weight=weights)

    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, coef=clf.coef_, intercept=clf.intercept_,
                        classes=clf.classes_.astype(np.int64), n_classes=n_classes,
                        feature_version=FEATURE_VERSION, **arrays)
    os.replace(tmp, out_path)
    return clf, vecs
