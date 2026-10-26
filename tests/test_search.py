def titles(hits):
    return [h.chunk.title for h in hits]


def test_hr_page_is_invisible_to_sales_even_for_an_exact_query(make_conduit):
    c = make_conduit()
    query = "Compensation Bands 2026 Confidential salary lakh"
    assert all("Compensation" not in t for t in titles(c.index.search(query, c.user("priya"), k=20)))
    assert "Compensation Bands 2026 (Confidential)" in titles(c.index.search(query, c.user("meera")))


def test_crm_is_invisible_to_support(make_conduit):
    c = make_conduit()
    hits = c.index.search("Northwind Bank annual recurring revenue renewal", c.user("arjun"), k=20)
    assert all(h.chunk.source != "crm" for h in hits)


def test_exact_identifiers_are_found_by_the_keyword_leg(make_conduit):
    c = make_conduit()
    top = c.index.search("TCK-502", c.user("arjun"), k=1)[0]
    assert top.chunk.doc_id == "helpdesk:TCK-502"
    assert top.keyword_rank == 0


def test_hybrid_combines_both_legs(make_conduit):
    c = make_conduit()
    hits = c.index.search("SSO certificate login failure", c.user("arjun"), k=3)
    assert hits[0].chunk.doc_id == "helpdesk:TCK-501"
    assert hits[0].keyword_rank is not None and hits[0].vector_rank is not None


def test_source_filter(make_conduit):
    c = make_conduit()
    hits = c.index.search("Northwind", c.user("anita"), sources={"helpdesk"})
    assert hits and all(h.chunk.source == "helpdesk" for h in hits)


def test_unknown_terms_return_nothing(make_conduit):
    c = make_conduit()
    assert c.index.search("zzqx flibbertigibbet", c.user("anita")) == []
