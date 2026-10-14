"""MCP auto-watch must run the shared post-processing pipeline (issue #1103, PR #1108)."""

from __future__ import annotations

import subprocess
import threading
from unittest.mock import Mock, patch

import pytest

from code_review_graph import incremental
from code_review_graph import main as crg_main
from code_review_graph.graph import GraphStore
from code_review_graph.postprocessing import run_post_processing


@pytest.fixture(autouse=True)
def isolate_mcp_state(monkeypatch):
    """Keep main() calls from leaking their default root or changing tool exposure."""
    monkeypatch.setattr(crg_main, "_default_repo_root", None)
    monkeypatch.delenv("CRG_TOOLS", raising=False)


def test_main_auto_watch_supplies_shared_postprocessing(tmp_path, monkeypatch):
    store = Mock(spec=GraphStore)
    start_watch = Mock(return_value=Mock(spec=threading.Thread))
    monkeypatch.setattr(crg_main, "get_db_path", Mock(return_value=tmp_path / "graph.db"))
    monkeypatch.setattr(crg_main, "GraphStore", Mock(return_value=store))
    monkeypatch.setattr(crg_main, "start_watch_thread", start_watch)
    monkeypatch.setattr(crg_main.mcp, "run", Mock())

    crg_main.main(repo_root=str(tmp_path), auto_watch=True)

    start_watch.assert_called_once_with(
        tmp_path, store, daemon=True, on_files_updated=run_post_processing,
    )
    store.close.assert_called_once_with()


def test_main_without_auto_watch_does_not_start_watcher(tmp_path, monkeypatch):
    open_store = Mock()
    start_watch = Mock()
    monkeypatch.setattr(crg_main, "GraphStore", open_store)
    monkeypatch.setattr(crg_main, "start_watch_thread", start_watch)
    monkeypatch.setattr(crg_main.mcp, "run", Mock())

    crg_main.main(repo_root=str(tmp_path), auto_watch=False)

    start_watch.assert_not_called()
    open_store.assert_not_called()


def test_start_watch_thread_forwards_callback_to_watch(tmp_path, monkeypatch):
    callback = Mock(return_value={})
    watch = Mock()
    monkeypatch.setattr(incremental, "watch", watch)

    with GraphStore(tmp_path / "graph.db") as store:
        thread = incremental.start_watch_thread(
            tmp_path, store, daemon=True, on_files_updated=callback,
        )
        assert thread is not None
        try:
            thread.join(timeout=5)
            assert not thread.is_alive(), "the mocked watch call did not finish"
            watch.assert_called_once_with(tmp_path, store, on_files_updated=callback)
        finally:
            thread.join(timeout=5)


def test_main_auto_watch_reconciliation_keeps_fts_consistent(tmp_path, monkeypatch):
    """Use real MCP wiring and startup reconciliation, with no native OS observer."""
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "mod.py"
    source.write_text(
        "def alphaHelper():\n    return 1\n\n"
        "def betaCaller():\n    return alphaHelper()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "mod.py"], cwd=repo, check=True, capture_output=True)
    db_path = incremental.get_db_path(repo)

    with GraphStore(db_path) as store:
        built = incremental.full_build(repo, store)
        assert built["errors"] == []
        processed = run_post_processing(store)
        assert not processed.get("warnings"), processed
        node_ids = {row[0] for row in store._conn.execute("SELECT id FROM nodes")}
        indexed_ids = {
            row[0] for row in store._conn.execute("SELECT id FROM nodes_fts_docsize")
        }
        assert indexed_ids == node_ids
        assert store._conn.execute(
            "SELECT COUNT(*) FROM nodes_fts f JOIN nodes n ON n.id = f.rowid "
            "WHERE nodes_fts MATCH ? AND n.name = ?",
            ("alphaHelper", "alphaHelper"),
        ).fetchone()[0] == 1

    source.write_text(
        source.read_text(encoding="utf-8")
        + "\ndef gammaHelper():\n    return alphaHelper() + 2\n",
        encoding="utf-8",
    )
    start_watch = incremental.start_watch_thread

    def start_and_join(*args, **kwargs):
        # Forward only what main supplies: never inject the missing callback here.
        thread = start_watch(*args, **kwargs)
        assert thread is not None
        thread.join(timeout=5)
        assert not thread.is_alive(), "startup reconciliation did not finish"
        return thread

    monkeypatch.setattr(crg_main, "start_watch_thread", start_and_join)
    monkeypatch.setattr(crg_main.mcp, "run", Mock())
    with (
        patch("watchdog.observers.Observer") as observer,
        patch("time.sleep", side_effect=KeyboardInterrupt),
    ):
        crg_main.main(repo_root=str(repo), auto_watch=True)
    observer.return_value.start.assert_called_once_with()

    with GraphStore(db_path) as store:
        assert "gammaHelper" in {node.name for node in store.get_nodes_by_file(str(source))}
        matches = {
            name: store._conn.execute(
                "SELECT COUNT(*) FROM nodes_fts f JOIN nodes n ON n.id = f.rowid "
                "WHERE nodes_fts MATCH ? AND n.name = ?",
                (name, name),
            ).fetchone()[0]
            for name in ("gammaHelper", "alphaHelper")
        }
        node_ids = {row[0] for row in store._conn.execute("SELECT id FROM nodes")}
        indexed_ids = {
            row[0] for row in store._conn.execute("SELECT id FROM nodes_fts_docsize")
        }
        assert matches == {"gammaHelper": 1, "alphaHelper": 1} and indexed_ids == node_ids, (
            f"matches={matches}; missing FTS IDs={sorted(node_ids - indexed_ids)}; "
            f"stale FTS IDs={sorted(indexed_ids - node_ids)}"
        )


def test_mcp_auto_watch_event_batches_keep_fts_consistent(tmp_path, monkeypatch):
    """MCP's callback must sync real FTS after each post-startup event batch."""
    from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileModifiedEvent

    from tests.test_watch_robustness import FakeObserver

    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "mod.py"
    source.write_text(
        "def alphaHelper():\n    return 1\n\n"
        "def betaCaller():\n    return 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "mod.py"], cwd=repo, check=True, capture_output=True)
    db_path = incremental.get_db_path(repo)

    def assert_fts_consistent(store, present, absent=()):
        node_ids = {row[0] for row in store._conn.execute("SELECT id FROM nodes")}
        # FTS is an external-content table: its docsize rows reveal the actual
        # indexed IDs, whereas a plain SELECT from nodes_fts reads live nodes.
        indexed_ids = {
            row[0] for row in store._conn.execute("SELECT id FROM nodes_fts_docsize")
        }
        assert not node_ids - indexed_ids, f"missing FTS IDs: {node_ids - indexed_ids}"
        assert not indexed_ids - node_ids, f"stale FTS IDs: {indexed_ids - node_ids}"
        for name in (*present, *absent):
            graph_ids = {
                row[0] for row in store._conn.execute(
                    "SELECT id FROM nodes WHERE name = ?", (name,),
                )
            }
            assert len(graph_ids) == (1 if name in present else 0), name
            matched_ids = {
                row[0] for row in store._conn.execute(
                    "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ?", (name,),
                )
            }
            assert matched_ids == graph_ids, f"{name}: FTS={matched_ids}, graph={graph_ids}"

    with GraphStore(db_path) as store:
        assert incremental.full_build(repo, store)["errors"] == []
        # Establish a healthy baseline only; subsequent batches must repair
        # FTS through the callback supplied by main(), with no test injection.
        processed = run_post_processing(store)
        assert not processed.get("warnings"), processed
        assert_fts_consistent(store, ("alphaHelper", "betaCaller"))

    observer = FakeObserver()
    ready = threading.Event()
    stop = threading.Event()
    threads = []
    start_watch = incremental.start_watch_thread

    def capture_thread(*args, **kwargs):
        thread = start_watch(*args, **kwargs)
        assert thread is not None
        threads.append(thread)
        return thread

    def wait_for_batches(_seconds):
        # The first loop tick occurs after reconciliation, handler.start(),
        # and observer.start(). Keep the watcher alive while MCP handles tests.
        ready.set()
        stop.wait(timeout=10)
        raise KeyboardInterrupt

    def exercise_batches(**kwargs):
        try:
            assert ready.wait(timeout=5), "watcher did not reach its post-startup loop"
            assert observer.started and observer.handler is not None
            handler = observer.handler
            assert handler.events_seen == 0
            with GraphStore(db_path) as store:
                assert_fts_consistent(store, ("alphaHelper", "betaCaller"))

                created = repo / "extra.py"
                created.write_text("def gammaHelper():\n    return 3\n", encoding="utf-8")
                # Reuse the synchronous batch boundary used by existing watcher
                # tests: process() returns after real updates and post-processing.
                handler.process([FileCreatedEvent(str(created))])
                handler.raise_if_failed()
                assert handler.events_seen == 1
                assert_fts_consistent(store, ("alphaHelper", "betaCaller", "gammaHelper"))

                source.write_text(
                    "def deltaHelper():\n    return 4\n\n"
                    "def betaCaller():\n    return 2\n",
                    encoding="utf-8",
                )
                handler.process([FileModifiedEvent(str(source))])
                handler.raise_if_failed()
                assert handler.events_seen == 2
                assert_fts_consistent(
                    store, ("deltaHelper", "betaCaller", "gammaHelper"), ("alphaHelper",),
                )

                created.unlink()
                handler.process([FileDeletedEvent(str(created))])
                handler.raise_if_failed()
                assert handler.events_seen == 3
                assert not store.get_nodes_by_file(str(created))
                assert_fts_consistent(
                    store, ("deltaHelper", "betaCaller"), ("alphaHelper", "gammaHelper"),
                )
        finally:
            stop.set()
            # main() closes its GraphStore as soon as the mocked transport exits.
            # Join first so neither watch nor its debouncer outlives that store.
            for thread in threads:
                thread.join(timeout=5)
                assert not thread.is_alive(), "watcher did not stop after event batches"

    monkeypatch.setattr(crg_main, "start_watch_thread", capture_thread)
    monkeypatch.setattr(crg_main.mcp, "run", exercise_batches)
    with (
        patch("watchdog.observers.Observer", return_value=observer),
        patch("time.sleep", side_effect=wait_for_batches),
    ):
        crg_main.main(repo_root=str(repo), auto_watch=True)
    assert observer.stopped


# ---------------------------------------------------------------------------
# Edge cases for #1108: restarts, stale graphs from earlier releases, failure
# paths, other entry points, large batches and awkward paths.
# ---------------------------------------------------------------------------

_ALPHA_BETA = (
    "def alphaHelper():\n    return 1\n\n"
    "def betaCaller():\n    return alphaHelper()\n"
)
_GAMMA = "\ndef gammaHelper():\n    return alphaHelper() + 2\n"


def _git_repo(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    return root


def _fts_drift(store):
    """Return (node IDs missing from FTS, FTS IDs with no node)."""
    node_ids = {row[0] for row in store._conn.execute("SELECT id FROM nodes")}
    indexed_ids = {row[0] for row in store._conn.execute("SELECT id FROM nodes_fts_docsize")}
    return node_ids - indexed_ids, indexed_ids - node_ids


def _valid_matches(store, name):
    return store._conn.execute(
        "SELECT COUNT(*) FROM nodes_fts f JOIN nodes n ON n.id = f.rowid "
        "WHERE nodes_fts MATCH ? AND n.name = ?",
        (name, name),
    ).fetchone()[0]


def _healthy_graph(repo):
    db_path = incremental.get_db_path(repo)
    with GraphStore(db_path) as store:
        assert incremental.full_build(repo, store)["errors"] == []
        processed = run_post_processing(store)
        assert not processed.get("warnings"), processed
        assert _fts_drift(store) == (set(), set())
    return db_path


def _serve_through_startup_reconciliation(monkeypatch, repo, **main_kwargs):
    """Run main(auto_watch=True) with the real watcher up to its first loop tick.

    The OS observer is a mock and the first tick interrupts the loop, so the
    thread ends right after startup reconciliation.  Returns the observer mock.
    """
    start_watch = incremental.start_watch_thread
    threads = []

    def start_and_join(*args, **kwargs):
        thread = start_watch(*args, **kwargs)
        assert thread is not None
        thread.join(timeout=30)
        assert not thread.is_alive(), "startup reconciliation did not finish"
        threads.append(thread)
        return thread

    serve = Mock()
    monkeypatch.setattr(crg_main, "start_watch_thread", start_and_join)
    monkeypatch.setattr(crg_main.mcp, "run", serve)
    with (
        patch("watchdog.observers.Observer") as observer,
        patch("time.sleep", side_effect=KeyboardInterrupt),
    ):
        crg_main.main(repo_root=str(repo), auto_watch=True, **main_kwargs)
    assert len(threads) == 1
    # The MCP transport still ran, whatever happened on the watcher thread.
    serve.assert_called_once()
    return observer


def test_restart_without_changes_skips_postprocessing(tmp_path, monkeypatch):
    """A healthy no-op start stays cheap and idempotent across restarts."""
    from code_review_graph import postprocessing

    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    repo = _git_repo(tmp_path / "repo", {"mod.py": _ALPHA_BETA})
    db_path = _healthy_graph(repo)
    spy = Mock(wraps=run_post_processing)
    monkeypatch.setattr(postprocessing, "run_post_processing", spy)

    for _ in range(2):
        observer = _serve_through_startup_reconciliation(monkeypatch, repo)
        observer.return_value.start.assert_called_once_with()

    spy.assert_not_called()
    with GraphStore(db_path) as store:
        assert _fts_drift(store) == (set(), set())
        assert _valid_matches(store, "alphaHelper") == 1


def test_first_change_repairs_index_left_stale_by_earlier_release(tmp_path, monkeypatch):
    """A graph a pre-fix auto-watch left stale is fully repaired by the next batch.

    The repair covers rows of files the batch did not touch, because the
    callback syncs FTS with no file hint and so compares the whole table.
    """
    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    repo = _git_repo(
        tmp_path / "repo",
        {"mod.py": _ALPHA_BETA, "other.py": "def otherHelper():\n    return 0\n"},
    )
    db_path = _healthy_graph(repo)
    module = repo / "mod.py"
    module.write_text(module.read_text(encoding="utf-8") + _GAMMA, encoding="utf-8")
    with GraphStore(db_path) as store:
        # What auto-watch before the fix did: update nodes, never touch FTS.
        updated = incremental.incremental_update(repo, store, changed_files=["mod.py"])
        assert updated["files_updated"] == 1
        missing, stale = _fts_drift(store)
        assert missing and stale
        assert _valid_matches(store, "gammaHelper") == 0

    (repo / "other.py").write_text(
        "def otherHelper():\n    return 0\n\ndef zetaHelper():\n    return 1\n",
        encoding="utf-8",
    )
    _serve_through_startup_reconciliation(monkeypatch, repo)

    with GraphStore(db_path) as store:
        assert _fts_drift(store) == (set(), set())
        for name in ("alphaHelper", "betaCaller", "gammaHelper", "otherHelper", "zetaHelper"):
            assert _valid_matches(store, name) == 1, name


def test_postprocess_warning_stops_watcher_loudly_and_server_keeps_serving(
    tmp_path, monkeypatch, caplog,
):
    import logging
    import sqlite3

    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    repo = _git_repo(tmp_path / "repo", {"mod.py": _ALPHA_BETA})
    _healthy_graph(repo)
    module = repo / "mod.py"
    module.write_text(module.read_text(encoding="utf-8") + _GAMMA, encoding="utf-8")
    caplog.set_level(logging.ERROR, logger="code_review_graph.incremental")

    with patch(
        "code_review_graph.search.update_fts_index",
        side_effect=sqlite3.OperationalError("forced FTS failure"),
    ):
        observer = _serve_through_startup_reconciliation(monkeypatch, repo)

    # The failed batch ends the watcher before it starts observing ...
    observer.assert_not_called()
    # ... and says so, rather than serving a frozen graph in silence.
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "code_review_graph.incremental"
    ]
    assert any(
        "Auto-watch for" in message
        and "post-processing reported warnings" in message
        and "forced FTS failure" in message
        for message in messages
    ), messages


def test_postprocess_crash_ends_only_the_watcher_thread_and_stays_off_stdout(
    tmp_path, monkeypatch, capsys,
):
    """Stdio MCP owns stdout: a crash on the watcher thread must not write there."""
    from code_review_graph import postprocessing

    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    repo = _git_repo(tmp_path / "repo", {"mod.py": _ALPHA_BETA})
    _healthy_graph(repo)
    module = repo / "mod.py"
    module.write_text(module.read_text(encoding="utf-8") + _GAMMA, encoding="utf-8")
    seen = []
    monkeypatch.setattr(threading, "excepthook", seen.append)
    monkeypatch.setattr(
        postprocessing, "_trace_flows", Mock(side_effect=ValueError("flow crash")),
    )

    observer = _serve_through_startup_reconciliation(monkeypatch, repo)

    observer.assert_not_called()
    assert [(hook.thread.name, type(hook.exc_value)) for hook in seen] == [
        ("crg-watch", ValueError),
    ]
    assert capsys.readouterr().out == ""


def test_start_watch_thread_keeps_positional_daemon_and_optional_callback(
    tmp_path, monkeypatch,
):
    """Callers written before #1108 keep working; the callback is keyword-only."""
    watch = Mock()
    monkeypatch.setattr(incremental, "watch", watch)

    with GraphStore(tmp_path / "graph.db") as store:
        thread = incremental.start_watch_thread(tmp_path, store, False)
        assert thread is not None
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert thread.daemon is False
        args, kwargs = watch.call_args
        assert args == (tmp_path, store)
        assert kwargs.get("on_files_updated") is None

        with pytest.raises(TypeError):
            incremental.start_watch_thread(tmp_path, store, True, Mock())
    watch.assert_called_once()


def test_http_auto_watch_supplies_shared_postprocessing(tmp_path, monkeypatch):
    store = Mock(spec=GraphStore)
    start_watch = Mock(return_value=Mock(spec=threading.Thread))
    serve = Mock()
    monkeypatch.setattr(crg_main, "get_db_path", Mock(return_value=tmp_path / "graph.db"))
    monkeypatch.setattr(crg_main, "GraphStore", Mock(return_value=store))
    monkeypatch.setattr(crg_main, "start_watch_thread", start_watch)
    monkeypatch.setattr(crg_main.mcp, "run", serve)

    crg_main.main(
        repo_root=str(tmp_path),
        auto_watch=True,
        transport="streamable-http",
        host="127.0.0.1",
        port=5555,
    )

    start_watch.assert_called_once_with(
        tmp_path, store, daemon=True, on_files_updated=run_post_processing,
    )
    assert serve.call_args.kwargs["transport"] == "streamable-http"
    store.close.assert_called_once_with()


@pytest.mark.parametrize("command", ["serve", "mcp"])
def test_cli_auto_watch_reaches_watcher_with_postprocessing(tmp_path, monkeypatch, command):
    """`serve --auto-watch` and its `mcp` alias both wire the callback end to end."""
    import sys

    from code_review_graph import cli

    store = Mock(spec=GraphStore)
    start_watch = Mock(return_value=Mock(spec=threading.Thread))
    monkeypatch.setattr(crg_main, "get_db_path", Mock(return_value=tmp_path / "graph.db"))
    monkeypatch.setattr(crg_main, "GraphStore", Mock(return_value=store))
    monkeypatch.setattr(crg_main, "start_watch_thread", start_watch)
    monkeypatch.setattr(crg_main.mcp, "run", Mock())
    monkeypatch.setattr(
        sys, "argv", ["code-review-graph", command, "--repo", str(tmp_path), "--auto-watch"],
    )

    cli.main()

    start_watch.assert_called_once()
    assert start_watch.call_args.kwargs["on_files_updated"] is run_post_processing
    store.close.assert_called_once_with()


def test_main_auto_watch_without_watchdog_still_serves(tmp_path, monkeypatch, caplog):
    import logging
    import sys

    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    serve = Mock()
    monkeypatch.setattr(crg_main.mcp, "run", serve)
    monkeypatch.setitem(sys.modules, "watchdog", None)
    caplog.set_level(logging.WARNING)

    crg_main.main(repo_root=str(tmp_path), auto_watch=True)

    serve.assert_called_once()
    assert "Auto-watch was requested but could not be started" in caplog.text


def test_large_batch_takes_fts_rebuild_path_and_stays_consistent(tmp_path, monkeypatch):
    """A batch past the delta threshold rebuilds FTS instead of applying a delta."""
    from code_review_graph import search

    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    repo = _git_repo(tmp_path / "repo", {"mod.py": _ALPHA_BETA})
    db_path = _healthy_graph(repo)
    # Startup reconciliation re-reads stored files, so grow the tracked one.
    (repo / "mod.py").write_text(
        _ALPHA_BETA + "".join(f"\ndef bulk{i}():\n    return {i}\n" for i in range(600)),
        encoding="utf-8",
    )
    rebuild = Mock(wraps=search.rebuild_fts_index)
    monkeypatch.setattr(search, "rebuild_fts_index", rebuild)

    _serve_through_startup_reconciliation(monkeypatch, repo)

    rebuild.assert_called()
    with GraphStore(db_path) as store:
        assert _fts_drift(store) == (set(), set())
        for name in ("alphaHelper", "bulk0", "bulk599"):
            assert _valid_matches(store, name) == 1, name


def test_reconciliation_with_unicode_and_spaces_in_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    repo = _git_repo(tmp_path / "répo dir", {"pkg dir/mód ule.py": _ALPHA_BETA})
    db_path = _healthy_graph(repo)
    module = repo / "pkg dir" / "mód ule.py"
    module.write_text(module.read_text(encoding="utf-8") + _GAMMA, encoding="utf-8")

    _serve_through_startup_reconciliation(monkeypatch, repo)

    with GraphStore(db_path) as store:
        assert "gammaHelper" in {node.name for node in store.get_nodes_by_file(str(module))}
        assert _fts_drift(store) == (set(), set())
        for name in ("alphaHelper", "betaCaller", "gammaHelper"):
            assert _valid_matches(store, name) == 1, name
