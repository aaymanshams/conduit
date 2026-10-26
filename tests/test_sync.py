import json

import pytest

from conduit.connectors.crm import CRMConnector
from conduit.embeddings import HashingEmbedder
from conduit.store import Store
from conduit.sync import sync_source


def test_first_sync_indexes_everything(make_conduit):
    c = make_conduit()
    assert c.store.counts() == {"crm": 4, "helpdesk": 4, "wiki": 4}
    assert len(c.index) >= 12


def test_resync_without_changes_embeds_nothing(make_conduit):
    c = make_conduit()
    stats = c.sync()
    assert sum(s.chunks_embedded for s in stats) == 0
    assert sum(s.added + s.updated + s.deleted for s in stats) == 0


def test_only_changed_records_are_re_embedded(make_conduit):
    c = make_conduit()
    path = c.crm.store.path
    accounts = json.loads(path.read_text())
    accounts[2]["notes"] = "Renewal signed for two more years after a 5 percent discount."
    accounts[2]["updated_at"] = "2026-10-26T10:00:00Z"
    path.write_text(json.dumps(accounts))

    (stats,) = c.sync("crm")
    assert (stats.updated, stats.added, stats.chunks_embedded) == (1, 0, 1)
    hits = c.index.search("renewal signed two more years", c.user("priya"), k=1)
    assert hits[0].chunk.doc_id == "crm:ACC-1003"


def test_deleted_records_leave_the_index(make_conduit):
    c = make_conduit()
    page = c.settings.data_dir / "sources" / "wiki" / "deployment-options.md"
    page.unlink()
    (stats,) = c.sync("wiki")
    assert stats.deleted == 1
    assert "wiki:deployment-options" not in c.store.doc_ids("wiki")
    assert all(h.chunk.doc_id != "wiki:deployment-options" for h in c.index.search("on-premises kubernetes", c.user("anita")))


def test_permission_change_alone_triggers_reindex(make_conduit):
    c = make_conduit()
    page = c.settings.data_dir / "sources" / "wiki" / "refund-policy.md"
    page.write_text(page.read_text().replace("groups: everyone", "groups: leadership"))
    (stats,) = c.sync("wiki")
    assert stats.updated == 1
    assert c.index.search("annual plans refunded", c.user("arjun")) == []


class FailingEmbedder(HashingEmbedder):
    def embed(self, texts):
        raise RuntimeError("provider down")


def test_failed_sync_does_not_advance_the_cursor(tmp_path, settings):
    store = Store(tmp_path / "t.db")
    connector = CRMConnector(settings.seed_dir / "crm_accounts.json")
    with pytest.raises(RuntimeError):
        sync_source(store, connector, FailingEmbedder())
    assert store.get_cursor("crm") is None
    stats = sync_source(store, connector, HashingEmbedder())
    assert stats.added == 4
    assert store.get_cursor("crm") == "2026-10-20T09:15:00Z"
