import re

# User-facing text lives here, so it can be translated in one place later.
NO_ANSWER = {
    "fa": "در اسناد شما پاسخی برای این سؤال پیدا نشد.",
    "en": "No answer to this question was found in your documents.",
}

_ARABIC_SCRIPT = re.compile("[؀-ۿ]")
_LATIN = re.compile("[A-Za-z]")


def language_of(text: str) -> str:
    # Majority script, so "What does فیدیبو mean?" stays English.
    return "fa" if len(_ARABIC_SCRIPT.findall(text)) > len(_LATIN.findall(text)) else "en"


def no_answer(question: str) -> str:
    return NO_ANSWER[language_of(question)]


_LABEL = re.compile(r"^\s*(no answer\s*:)?\s*", re.IGNORECASE)


def is_no_answer(text: str) -> bool:
    # Either language counts, and a copied "No answer:" label is tolerated: the model does both.
    return _LABEL.sub("", text).strip() in NO_ANSWER.values()
