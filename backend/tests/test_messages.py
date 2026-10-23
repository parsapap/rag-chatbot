import pytest

from app.core.messages import NO_ANSWER, is_no_answer, language_of, no_answer


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("فیدیبو آرت چیست؟", "fa"),
        ("What is Fidibo Art?", "en"),
        ("What does فیدیبو mean?", "en"),
        ("موجودی نقد در 1404/09/30 چقدر بود؟", "fa"),
        ("1404?", "en"),
    ],
)
def test_language_of_uses_majority_script(text, lang):
    assert language_of(text) == lang


def test_no_answer_matches_question_language():
    assert no_answer("پایتخت فرانسه کجاست؟") == NO_ANSWER["fa"]
    assert no_answer("What is the capital of France?") == NO_ANSWER["en"]


def test_both_sentences_count_as_no_answer():
    assert is_no_answer(NO_ANSWER["fa"])
    assert is_no_answer(f"  {NO_ANSWER['en']}\n")
    assert not is_no_answer("Fidibo Art started in 1403 [1].")


def test_copied_label_still_counts_as_no_answer():
    assert is_no_answer(f"No answer: {NO_ANSWER['fa']}")
    assert is_no_answer(f"no answer:{NO_ANSWER['en']}")
