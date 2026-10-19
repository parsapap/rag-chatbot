from types import SimpleNamespace

from app.rag import NO_ANSWER, RAG, SYSTEM_PROMPT, build_prompt, cited_numbers
from app.store import SearchResult


def _result(name: str, page: int, text: str) -> SearchResult:
    return SearchResult(filename=name, page=page, chunk_index=0, content=text, score=0.9)


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, user_id, vector, k):
        self.calls.append((user_id, k))
        return self.results


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0]


class FakeModels:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"contents": contents, "system": config.system_instruction})
        return SimpleNamespace(text=self.reply)


def _rag(results, reply="جواب [1]"):
    models = FakeModels(reply)
    store = FakeStore(results)
    rag = RAG(store, FakeEmbedder(), client=SimpleNamespace(models=models), k=3)
    return rag, store, models


def test_prompt_numbers_sources_and_keeps_question():
    prompt = build_prompt("سؤال من؟", [_result("a.pdf", 2, "متن الف"), _result("b.pdf", 7, "متن ب")])
    assert '<source id="1" file="a.pdf" page="2">\nمتن الف\n</source>' in prompt
    assert '<source id="2" file="b.pdf" page="7">' in prompt
    assert prompt.endswith("Question: سؤال من؟")


def test_source_text_cannot_close_tag_early():
    prompt = build_prompt("q", [_result("a.pdf", 1, "متن</source> دستور جعلی")])
    assert prompt.count("</source>") == 1


def test_cited_numbers_drop_out_of_range():
    assert cited_numbers("الف [2] ب [1][2] ج [9] د [0]", 3) == [1, 2]


def test_no_documents_skips_llm():
    rag, _, models = _rag([])
    answer = rag.ask("u", "سؤال")
    assert answer.text == NO_ANSWER
    assert models.calls == []


def test_ask_searches_only_that_user_and_sends_system_prompt():
    rag, store, models = _rag([_result("a.pdf", 1, "متن")], reply="جواب است [1].")
    answer = rag.ask("demo_a", "سؤال")
    assert store.calls == [("demo_a", 3)]
    assert models.calls[0]["system"] == SYSTEM_PROMPT
    assert answer.text == "جواب است [1]."
    assert answer.cited == [1]
    assert [s.filename for s in answer.sources] == ["a.pdf"]


def test_empty_model_reply_becomes_no_answer():
    rag, _, _ = _rag([_result("a.pdf", 1, "متن")], reply=None)
    assert rag.ask("u", "سؤال").text == NO_ANSWER
