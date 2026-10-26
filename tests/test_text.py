from conduit.search.text import chunk_text, tokenize


def test_tokenize_lowercases_drops_stopwords_and_folds_plurals():
    assert tokenize("The Tickets for our Policies are OPEN") == ["ticket", "policy", "open"]


def test_short_paragraphs_are_packed_together():
    assert chunk_text("one.\n\ntwo.\n\nthree.", max_chars=100) == ["one.\n\ntwo.\n\nthree."]


def test_chunks_respect_the_size_limit_and_keep_all_text():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_long_paragraph_is_split_on_sentences():
    para = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(para, max_chars=200)
    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)
