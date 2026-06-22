"""
Test cho Rule Scorer (emotion_mlp.rule_score) và xử lý phủ định (_is_negated).
Chỉ nhắm vào phần logic cố định (không phụ thuộc trạng thái online learning) -
LEXICON tĩnh, NEGATIONS, scoring - để bắt sớm các regression như lỗi v3.6
(phủ định không áp dụng cho bigram, không nhận diện phủ định nhiều từ) và
lỗi "vocab chết" trước v4.0 (cụm cảm xúc 3-5 từ trong LEXICON không bao giờ
được tokenizer/rule_score cũ nhận diện thành 1 cụm).
"""
import numpy as np
import pytest

import emotion_mlp
from emotion_mlp import rule_score, _is_negated, _tokenize
from lexicon import EMOTIONS, NEGATIONS

N = len(EMOTIONS)
HAPPY = EMOTIONS.index("happy")
SAD = EMOTIONS.index("sad")
FOCUSED = EMOTIONS.index("focused")
ANGRY = EMOTIONS.index("angry")
STRESSED = EMOTIONS.index("stressed")
ENERGETIC = EMOTIONS.index("energetic")


def is_uniform(scores):
    return np.allclose(scores, 1.0 / N)


class TestIsNegated:
    def test_no_words_before_is_not_negated(self):
        assert _is_negated(["vui"], 0) is False

    @pytest.mark.parametrize("neg_word", [
        "không", "chẳng", "chả", "đâu", "chưa", "khỏi", "ko", "k",
    ])
    def test_single_word_negation(self, neg_word):
        words = [neg_word, "vui"]
        assert _is_negated(words, 1) is True

    @pytest.mark.parametrize("phrase", [
        "không hề", "chẳng hề", "chả hề", "chưa hề",
        "không phải", "chẳng phải", "chả phải",
        "không có", "đâu có", "có đâu", "đâu phải",
    ])
    def test_two_word_negation(self, phrase):
        words = phrase.split() + ["vui"]
        assert _is_negated(words, len(words) - 1) is True

    @pytest.mark.parametrize("phrase", [
        "không bao giờ", "chẳng bao giờ", "chả bao giờ", "chưa bao giờ",
    ])
    def test_three_word_negation(self, phrase):
        words = phrase.split() + ["vui"]
        assert _is_negated(words, len(words) - 1) is True

    def test_unrelated_word_before_is_not_negated(self):
        assert _is_negated(["rất", "vui"], 1) is False

    def test_negation_must_be_immediately_before(self):
        # "không" cách "vui" 2 từ ở giữa ("phải", "vậy") -> không tính là phủ định
        words = ["không", "phải", "vậy", "vui"]
        assert _is_negated(words, 3) is False


class TestRuleScoreBasics:
    def test_positive_sentence_picks_matching_emotion(self):
        scores = rule_score("hôm nay tôi rất vui")
        assert np.argmax(scores) == HAPPY
        assert not is_uniform(scores)

    def test_bigram_match_picks_matching_emotion(self):
        scores = rule_score("tôi tập trung học bài")
        assert np.argmax(scores) == FOCUSED
        assert not is_uniform(scores)

    def test_no_lexicon_match_is_uniform(self):
        scores = rule_score("asdkj qwoeiqwe random text 12345")
        assert is_uniform(scores)


class TestRuleScoreNegation:
    """Regression cho v3.6: trước đây phủ định chỉ áp dụng cho unigram, và
    NEGATIONS chỉ có từ đơn -> các câu dưới đây từng bị nhận sai thành
    happy/focused thay vì bị triệt tiêu điểm (uniform)."""

    def test_single_word_negation_cancels_unigram(self):
        scores = rule_score("tôi không vui")
        assert is_uniform(scores)

    def test_two_word_negation_cancels_unigram(self):
        scores = rule_score("tôi không hề vui")
        assert is_uniform(scores)

    def test_three_word_negation_cancels_unigram(self):
        scores = rule_score("tôi chẳng bao giờ vui khi ở đây")
        assert is_uniform(scores)

    def test_negation_cancels_bigram(self):
        scores = rule_score("tôi không tập trung được")
        assert is_uniform(scores)

    def test_three_word_negation_cancels_bigram(self):
        scores = rule_score("cô ấy không bao giờ tập trung")
        assert is_uniform(scores)

    def test_negation_only_cancels_negated_emotion(self):
        # "không" phủ định "buồn" (sad) nhưng không ảnh hưởng đến "vui" (happy)
        scores = rule_score("tôi vui chứ không buồn")
        assert np.argmax(scores) == HAPPY
        assert not is_uniform(scores)
        assert scores[SAD] < scores[HAPPY]

    def test_negation_word_far_from_target_has_no_effect(self):
        # "không phải" đứng cách "vui" nhiều từ -> không được tính là phủ định
        scores = rule_score("không phải vì vậy mà tôi vui")
        assert np.argmax(scores) == HAPPY
        assert not is_uniform(scores)

    def test_negation_after_phrase_does_not_apply_backwards(self):
        # Phủ định đứng SAU cụm "tập trung" không ảnh hưởng ngược lại cụm đó
        scores = rule_score("tập trung học bài không phải sở thích của tôi")
        assert np.argmax(scores) == FOCUSED
        assert not is_uniform(scores)


class TestWordSegmentationModel3:
    """Regression cho v4.0 (Model 3): trước đây _tokenize()/rule_score() chỉ
    thử ghép đúng 2 từ liền kề (bigram-only) -> các cụm cảm xúc dài hơn trong
    LEXICON (3-5 từ, ví dụ "không thể chấp nhận được" dưới "angry") có
    embedding/entry sẵn nhưng KHÔNG BAO GIỜ được nhận diện thành 1 cụm -
    "vocab chết". Generalize sang longest-match N-gram sửa lỗi này."""

    @pytest.mark.parametrize("text,expected", [
        ("không thể chấp nhận được", ANGRY),       # 5 từ
        ("ghét cay ghét đắng", ANGRY),              # 4 từ
        ("tràn đầy năng lượng", ENERGETIC),         # 4 từ
        ("áp lực nặng", STRESSED),                  # 3 từ
        ("quá tải tinh thần", STRESSED),            # 4 từ
    ])
    def test_long_phrase_now_matches(self, text, expected):
        scores = rule_score(text)
        assert np.argmax(scores) == expected
        assert not is_uniform(scores)

    def test_long_phrase_tokenizes_as_single_token(self):
        # Nhánh MLP phải nhìn thấy đúng 1 token cho cả cụm, không bị tách
        # thành các âm tiết rời rạc rồi rơi mất (id=None) như tokenizer cũ.
        toks = _tokenize("không thể chấp nhận được")
        assert len(toks) == 1
        text, tid = toks[0]
        assert text == "không thể chấp nhận được"
        assert tid is not None

    def test_negation_cancels_long_phrase(self):
        scores = rule_score("tôi không hề tràn đầy năng lượng")
        assert is_uniform(scores)

    def test_all_negations_are_registered_in_vocab(self):
        # v4.0: NEGATIONS phải có mặt trong VOCAB_IDX một cách CHẮC CHẮN
        # (đăng ký thẳng lúc khởi động), không còn phụ thuộc việc từ đó có
        # "tình cờ" xuất hiện gần một từ cảm xúc trong feedback hay chưa.
        for neg in NEGATIONS:
            assert neg in emotion_mlp.VOCAB_IDX, f"{neg!r} chưa có trong VOCAB_IDX"
