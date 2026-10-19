import pymupdf
import pytest

from app.loaders import LoaderError, _fix_lam_alef, load, normalize, strip_page_furniture

ZWNJ = "\u200c"


def test_arabic_letters_become_persian():
    assert normalize("\u0643تاب عل\u064a") == "کتاب علی"


def test_persian_and_arabic_digits_become_ascii():
    assert normalize("ماده \u06f1\u06f2 و \u0663\u0664") == "ماده 12 و 34"


def test_zwnj_is_kept_inside_words():
    assert normalize(f"می{ZWNJ}خواهم") == f"می{ZWNJ}خواهم"


def test_zwnj_next_to_space_becomes_space():
    assert normalize(f"کتاب {ZWNJ}ها") == "کتاب ها"


def test_invisible_marks_removed():
    assert normalize("\ufeffسل\u0640\u0640\u0640ام\u200f") == "سلام"


def test_whitespace_collapsed():
    assert normalize("  a   b \n\n\n\n c  ") == "a b\n\nc"


def test_load_txt(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("\ufeffسلام دن\u064aا", encoding="utf-8")
    [page] = load(f)
    assert page.text == "سلام دنیا"
    assert page.source == "note.txt"
    assert page.page == 1


def test_load_md(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nBody text", encoding="utf-8")
    [page] = load(f)
    assert page.text == "# Title\n\nBody text"


def test_non_utf8_rejected(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_bytes("سلام".encode("cp1256"))
    with pytest.raises(LoaderError, match="UTF-8"):
        load(f)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "file.docx"
    f.write_text("x")
    with pytest.raises(LoaderError, match="unsupported"):
        load(f)


def test_missing_file(tmp_path):
    with pytest.raises(LoaderError, match="not found"):
        load(tmp_path / "nope.pdf")


def _make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    return doc


def test_load_pdf_keeps_page_numbers(tmp_path):
    f = tmp_path / "paper.pdf"
    doc = _make_pdf(f, ["First page", "", "Third page"])
    doc.save(f)
    pages = load(f)
    assert [p.page for p in pages] == [1, 2, 3]
    assert pages[0].text == "First page"
    assert pages[1].text == ""
    assert pages[2].text == "Third page"


def test_encrypted_pdf_rejected(tmp_path):
    f = tmp_path / "locked.pdf"
    doc = _make_pdf(f, ["secret"])
    doc.save(f, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(LoaderError, match="encrypted"):
        load(f)


def test_corrupt_pdf_rejected(tmp_path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"not a pdf at all")
    with pytest.raises(LoaderError, match="corrupt"):
        load(f)


def _chars(spec):
    # spec: list of (char, width); x positions don't matter for the fix.
    return [{"c": c, "bbox": (0, 0, w, 10)} for c, w in spec]


def test_zero_width_alef_lam_is_ligature():
    chars = _chars([("ط", 5), ("و", 5), ("ا", 0), ("ل", 6), ("ن", 5), ("ی", 5)])
    assert _fix_lam_alef(chars) == "طولانی"


def test_real_alef_lam_is_kept():
    chars = _chars([("ا", 2), ("ل", 4), ("ز", 4), ("ا", 2), ("م", 5)])
    assert _fix_lam_alef(chars) == "الزام"


def test_persian_number_separators():
    assert normalize("778،910 و 1٫65") == "778,910 و 1.65"


def test_arabic_comma_in_prose_is_kept():
    assert normalize("سلام، 12، 3") == "سلام، 12، 3"


def test_running_header_and_page_numbers_removed():
    topics = ["کتاب صوتی", "فروش سالانه", "پادکست جدید", "کنسرت فیلم", "تئاتر شهر", "ژانر رمان"]
    texts = []
    for i, topic in enumerate(topics):
        lines = [str(349 + i)] if i % 2 == 0 else [f"شرکت‌های دیگر گروه دیجی‌کالا فصل آخر{349 + i}"]
        texts.append("\n".join(lines + [f"درباره {topic}", "درصد 20"]))
    out = strip_page_furniture(texts)
    for topic, t in zip(topics, out):
        assert t == f"درباره {topic}\nدرصد 20"


def test_repeated_line_without_number_is_kept():
    texts = [f"(مبالغ به میلیون ریال)\nردیف {i}\n{100 + i}" for i in range(5)]
    out = strip_page_furniture(texts)
    assert all(t.startswith("(مبالغ به میلیون ریال)") for t in out)


def test_numbers_that_are_data_not_removed():
    texts = [f"مبلغ کل\n{n}" for n in (500, 72, 9100, 31)]
    assert strip_page_furniture(texts) == texts
