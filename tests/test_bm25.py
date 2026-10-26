import random

import pytest

from conduit.search import bm25
from conduit.search.bm25 import PyBM25Index

VOCAB = [f"w{i}" for i in range(40)]


def build(cls, docs):
    idx = cls()
    for doc_id, tokens in docs.items():
        idx.add(doc_id, tokens)
    return idx


def random_corpus(seed, n=200):
    rng = random.Random(seed)
    return {i * 7 + 3: [rng.choice(VOCAB) for _ in range(rng.randint(1, 40))] for i in range(n)}


def test_scores_and_ranks_matching_documents():
    idx = build(PyBM25Index, {1: ["refund", "policy", "annual"], 2: ["sso", "outage"], 3: ["refund", "refund", "credit"]})
    results = idx.search(["refund"], 10, [1, 2, 3])
    assert [d for d, _ in results] == [3, 1]  # higher term frequency ranks first
    assert all(score > 0 for _, score in results)


def test_disallowed_documents_are_never_returned():
    idx = build(PyBM25Index, {1: ["salary", "band"], 2: ["refund"]})
    assert idx.search(["salary"], 10, [2]) == []


def test_duplicate_query_terms_count_once():
    idx = build(PyBM25Index, {1: ["sso"], 2: ["other"]})
    assert idx.search(["sso", "sso"], 5, [1, 2]) == idx.search(["sso"], 5, [1, 2])


def test_duplicate_doc_id_is_rejected():
    idx = PyBM25Index()
    idx.add(1, ["a"])
    with pytest.raises(ValueError):
        idx.add(1, ["b"])


@pytest.mark.skipif(bm25.BACKEND != "cpp", reason="C++ extension not built")
@pytest.mark.parametrize("seed", range(5))
def test_cpp_and_python_backends_agree(seed):
    from conduit._bm25 import BM25Index as CppIndex

    docs = random_corpus(seed)
    rng = random.Random(seed + 100)
    allowed = rng.sample(sorted(docs), 150)
    cpp, py = build(CppIndex, docs), build(PyBM25Index, docs)
    for _ in range(20):
        query = [rng.choice(VOCAB) for _ in range(rng.randint(1, 5))]
        a, b = cpp.search(query, 10, allowed), py.search(query, 10, allowed)
        assert [d for d, _ in a] == [d for d, _ in b]
        assert [s for _, s in a] == pytest.approx([s for _, s in b])


@pytest.mark.skipif(bm25.BACKEND != "cpp", reason="C++ extension not built")
def test_cpp_backend_rejects_duplicate_ids():
    from conduit._bm25 import BM25Index as CppIndex

    idx = CppIndex()
    idx.add(1, ["a"])
    with pytest.raises(ValueError):
        idx.add(1, ["b"])
