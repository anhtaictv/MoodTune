"""
TF-IDF n-gram + Logistic Regression cho cảm xúc tiếng Việt (v5.1, sklearn trực tiếp).

4 kênh đặc trưng, mỗi kênh TF-IDF (sublinear tf, chuẩn hoá L2) riêng rồi nối lại:
  word        : n-gram 1-2 âm tiết trên câu đã chuẩn hoá (vi_text.normalize)
  char        : n-gram ký tự 2-5 trong phạm vi từ (char_wb) - chịu được lỗi chính tả
  noacc       : như char nhưng trên bản bỏ dấu  -> nhận ra câu gõ không dấu
  noacc_word  : như word nhưng trên bản bỏ dấu

Cả train lẫn inference đều là một sklearn Pipeline, lưu bằng joblib.
"""
import os
from functools import partial

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from vi_text import normalize, strip_accents

MODEL_FILE = "tfidf_model.joblib"
# Tăng khi đổi vi_text.normalize / TEENCODE / analyzer: model cũ trên server bị
# bỏ qua và train lại, thay vì âm thầm dùng vocab không còn khớp tiền xử lý.
FEATURE_VERSION = 2
CHANNELS = ("word", "char", "noacc", "noacc_word")
LR_C = 10.0
WORD_NGRAM = (1, 2)  # 5-fold CV: (1,3) không tốt hơn
NOACC_AUG_WEIGHT = 0.5   # trọng số bản sao bỏ dấu của mỗi câu train
PSEUDO_WEIGHT = 0.3      # trọng số câu chỉ có nhãn do model cũ đoán (type=predict)


def _channel_text(name, text):
    t = normalize(text)
    return strip_accents(t) if name.startswith("noacc") else t


def build_pipeline(c=LR_C, word_ngram=WORD_NGRAM):
    vecs = []
    for ch in CHANNELS:
        pre = partial(_channel_text, ch)  # partial của hàm module-level -> pickle được
        if ch.endswith("word"):
            vec = TfidfVectorizer(preprocessor=pre, ngram_range=word_ngram, sublinear_tf=True,
                                  token_pattern=r"\S+")
        else:
            vec = TfidfVectorizer(preprocessor=pre, analyzer="char_wb", ngram_range=(2, 5),
                                  sublinear_tf=True, min_df=2)
        vecs.append((ch, vec))
    return Pipeline([
        ("features", FeatureUnion(vecs)),
        ("clf", LogisticRegression(C=c, max_iter=3000, class_weight="balanced")),
    ])


class TfidfEmotionModel:
    """Bọc Pipeline: trả vector xác suất đủ n_classes kể cả khi train thiếu lớp."""

    def __init__(self, pipeline, n_classes, feature_version=FEATURE_VERSION):
        self.pipeline = pipeline
        self.n_classes = n_classes
        self.feature_version = feature_version

    @property
    def n_features(self):
        return len(self.pipeline.named_steps["clf"].coef_[0])

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        return cls(d["pipeline"], d["n_classes"], d.get("feature_version", 0))

    def save(self, path):
        # Ghi file tạm rồi rename -> backend không bao giờ đọc phải file ghi dở
        tmp = path + ".tmp"
        joblib.dump({"pipeline": self.pipeline, "n_classes": self.n_classes,
                     "feature_version": self.feature_version}, tmp)
        os.replace(tmp, path)

    def predict_proba(self, text):
        """Xác suất (n_classes,) theo thứ tự EMOTIONS."""
        full = np.zeros(self.n_classes)
        full[self.pipeline.classes_] = self.pipeline.predict_proba([text])[0]
        return full


def augment(texts, labels, weights):
    # Thêm bản sao bỏ dấu: dữ liệu train toàn câu có dấu nhưng người dùng
    # hay gõ không dấu -> kênh word/char có dấu cũng thấy dạng không dấu.
    texts = list(texts) + [strip_accents(normalize(t)) for t in texts]
    labels = list(labels) * 2
    weights = np.concatenate([weights, np.asarray(weights) * NOACC_AUG_WEIGHT])
    return texts, labels, weights


def train(texts, labels, weights, n_classes, out_path=None):
    texts, labels, weights = augment(texts, labels, weights)
    pipe = build_pipeline()
    pipe.fit(texts, labels, clf__sample_weight=weights)
    model = TfidfEmotionModel(pipe, n_classes)
    if out_path:
        model.save(out_path)
    return model
