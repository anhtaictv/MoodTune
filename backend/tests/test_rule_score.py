"""
Test cho Rule Scorer (emotion_mlp.rule_score) và xử lý phủ định (_is_negated).
Chỉ nhắm vào phần logic cố định (không phụ thuộc trạng thái online learning) -
LEXICON tĩnh, NEGATIONS, scoring - để bắt sớm các regression như lỗi v3.6
(phủ định không áp dụng cho bigram, không nhận diện phủ định nhiều từ).
"""
import numpy as np
import pytest

from emotion_mlp import rule_score, _is_negated
from lexicon import EMOTIONS

N = len(EMOTIONS)
HAPPY = EMOTIONS.index("happy")
SAD = EMOTIONS.index("sad")
FOCUSED = EMOTIONS.index("focused")


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
