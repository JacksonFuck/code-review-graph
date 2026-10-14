"""The README's OrcaRouter recipe drives the existing OpenAI-compatible provider.

PR #923 documents OrcaRouter as an OpenAI-compatible ``/v1/embeddings``
endpoint reached through ``--provider openai``. It ships no code, so these
tests pin the recipe itself: every variable it exports is one the openai
provider reads and the README documents, the command it prints parses, the
request it produces is the plain OpenAI wire format, and the vectors it
stores are partitioned from any other endpoint's.

Nothing here contacts a real endpoint. Wire tests use a patched ``urlopen``
or a fake gateway bound to 127.0.0.1.
"""

from __future__ import annotations

import json
import re
import shlex
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_review_graph.cli import build_parser
from code_review_graph.embeddings import (
    EmbeddingStore,
    OpenAIEmbeddingProvider,
    get_provider,
)
from code_review_graph.graph import GraphNode

ROOT = Path(__file__).parents[1]
README = ROOT / "README.md"
EMBEDDINGS_SOURCE = ROOT / "code_review_graph" / "embeddings.py"

RECIPE_KEY = "orca-test-key"
RECIPE_MODEL = "openai/text-embedding-3-small"
RECIPE_BASE_URL = "https://api.orcarouter.ai/v1"
RECIPE_VARS = {
    "CRG_OPENAI_BASE_URL",
    "CRG_OPENAI_API_KEY",
    "CRG_OPENAI_MODEL",
    "CRG_ACCEPT_CLOUD_EMBEDDINGS",
}
# Variables that change what the openai provider does. Cleared before each
# test so a developer's shell cannot leak into the recipe under test.
_PROVIDER_ENV = (
    "CRG_OPENAI_BASE_URL",
    "CRG_OPENAI_API_KEY",
    "CRG_OPENAI_MODEL",
    "CRG_OPENAI_DIMENSION",
    "CRG_OPENAI_BATCH_SIZE",
    "CRG_ACCEPT_CLOUD_EMBEDDINGS",
    "ORCAROUTER_API_KEY",
)


def _recipe_block() -> str:
    """Return the fenced block that follows the README's OrcaRouter link."""
    text = README.read_text(encoding="utf-8")
    anchor = text.find("(https://www.orcarouter.ai)")
    assert anchor != -1, "README no longer links OrcaRouter"
    match = re.compile(r"```bash\n(?P<body>.*?)\n```", re.DOTALL).search(text, anchor)
    assert match is not None, "no fenced bash block after the OrcaRouter link"
    return match.group("body")


def _recipe() -> tuple[dict[str, str], list[list[str]]]:
    """Split the recipe into its exports and its commands."""
    exports: dict[str, str] = {}
    commands: list[list[str]] = []
    for line in _recipe_block().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("export "):
            key, _, value = line[len("export "):].partition("=")
            exports[key] = value
        else:
            commands.append(shlex.split(line))
    return exports, commands


def _apply_recipe(monkeypatch, *, key: str = RECIPE_KEY, **overrides: str | None) -> None:
    """Export the recipe as a shell would, with ``$ORCAROUTER_API_KEY`` = ``key``."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    exports, _ = _recipe()
    for name, value in exports.items():
        value = value.replace("$ORCAROUTER_API_KEY", key)
        monkeypatch.setenv(name, value)
    for name, value in overrides.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _node(name: str, qualified: str, doc: str = "") -> GraphNode:
    return GraphNode(
        id=0, kind="Function", name=name, qualified_name=qualified,
        file_path="pkg/mod.py", line_start=1, line_end=2, language="python",
        parent_name=None, params=None, return_type=None, is_test=False,
        file_hash=None, extra={"docstring": doc} if doc else {},
    )


def _ok_response(vectors: list[list[float]]) -> MagicMock:
    body = json.dumps({
        "data": [{"embedding": v, "index": i} for i, v in enumerate(vectors)],
    }).encode("utf-8")
    response = MagicMock()
    response.read.return_value = body
    response.__enter__ = lambda self: self
    response.__exit__ = MagicMock(return_value=False)
    return response


@pytest.fixture
def fake_gateway():
    """An OpenAI-compatible gateway on loopback that checks the bearer key.

    Vectors are one-hot on a keyword so similarity ranking is exact.
    """
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            requests.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "payload": payload,
            })
            if self.headers.get("Authorization") != f"Bearer {RECIPE_KEY}":
                status = 401
                response = {"error": {"message": "invalid api key"}}
            else:
                status = 200
                response = {
                    "model": payload["model"],
                    "data": [
                        {"embedding": self._vector(text), "index": index}
                        for index, text in enumerate(payload["input"])
                    ],
                }
            body = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @staticmethod
        def _vector(text: str) -> list[float]:
            lowered = text.lower()
            if "password" in lowered:
                return [1.0, 0.0, 0.0]
            if "invoice" in lowered:
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield port, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestRecipeText:
    def test_recipe_exports_exactly_the_openai_provider_variables(self):
        exports, _ = _recipe()
        assert set(exports) == RECIPE_VARS
        assert exports["CRG_OPENAI_BASE_URL"] == RECIPE_BASE_URL
        assert exports["CRG_OPENAI_MODEL"] == RECIPE_MODEL
        assert exports["CRG_OPENAI_API_KEY"] == "$ORCAROUTER_API_KEY"
        assert exports["CRG_ACCEPT_CLOUD_EMBEDDINGS"] == "1"

    def test_every_recipe_variable_is_read_by_the_provider_and_documented(self):
        source = EMBEDDINGS_SOURCE.read_text(encoding="utf-8")
        readme = README.read_text(encoding="utf-8")
        for name in RECIPE_VARS:
            assert f'"{name}"' in source, f"{name} is not read by embeddings.py"
            assert f"| `{name}` |" in readme, f"{name} missing from the README env table"

    def test_recipe_command_is_a_valid_embed_invocation(self):
        _, commands = _recipe()
        assert len(commands) == 1
        argv = commands[0]
        assert argv[0] == "code-review-graph"
        args = build_parser().parse_args(argv[1:])
        assert args.command == "embed"
        assert args.provider == "openai"
        assert args.model is None  # the model comes from CRG_OPENAI_MODEL


class TestRecipeProviderResolution:
    def test_recipe_resolves_openai_provider_with_endpoint_identity(
        self, monkeypatch, capsys,
    ):
        _apply_recipe(monkeypatch)
        provider = get_provider("openai")
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.name == f"openai:{RECIPE_MODEL}@{RECIPE_BASE_URL}"
        assert RECIPE_KEY not in provider.name
        captured = capsys.readouterr()
        # The accept flag in the recipe silences the egress warning, and
        # nothing may reach stdout (MCP stdio transport).
        assert captured.err == ""
        assert captured.out == ""

    def test_recipe_is_treated_as_cloud_egress_without_the_accept_flag(
        self, monkeypatch, capsys,
    ):
        _apply_recipe(monkeypatch, CRG_ACCEPT_CLOUD_EMBEDDINGS=None)
        get_provider("openai")
        captured = capsys.readouterr()
        assert "'openai' cloud provider" in captured.err
        assert captured.out == ""

    def test_omitted_provider_selects_the_recipe_endpoint(self, monkeypatch):
        """MCP search omits ``provider``; the recipe must still be used (#551)."""
        _apply_recipe(monkeypatch)
        provider = get_provider(None)
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.name == f"openai:{RECIPE_MODEL}@{RECIPE_BASE_URL}"

    @pytest.mark.parametrize("key", ["", None])
    def test_unset_orcarouter_key_names_the_missing_variable(self, monkeypatch, key):
        # ``export CRG_OPENAI_API_KEY=$ORCAROUTER_API_KEY`` with the source
        # variable unset exports an empty string.
        _apply_recipe(monkeypatch, key="", CRG_OPENAI_API_KEY=key)
        with pytest.raises(ValueError, match="CRG_OPENAI_API_KEY"):
            get_provider("openai")

    def test_unset_orcarouter_key_is_a_structured_embed_graph_error(
        self, monkeypatch, tmp_path,
    ):
        from code_review_graph.tools import docs as docs_module

        _apply_recipe(monkeypatch, key="")
        (tmp_path / ".git").mkdir()
        result = docs_module.embed_graph(repo_root=str(tmp_path), provider="openai")
        assert result["status"] == "error"
        assert "CRG_OPENAI_API_KEY" in result["error"]


class TestRecipeWireFormat:
    def test_recipe_sends_the_plain_openai_request(self, monkeypatch):
        _apply_recipe(monkeypatch)
        provider = get_provider("openai")
        with patch(
            "urllib.request.urlopen",
            return_value=_ok_response([[0.1] * 4, [0.2] * 4]),
        ) as urlopen:
            vectors = provider.embed(["hash password", "naïve façade ✓"])

        assert len(vectors) == 2
        request = urlopen.call_args[0][0]
        assert request.full_url == f"{RECIPE_BASE_URL}/embeddings"
        assert request.get_method() == "POST"
        assert request.headers["Authorization"] == f"Bearer {RECIPE_KEY}"
        # Exactly model + input: no ``dimensions`` unless the user pins one,
        # and the slash-prefixed model id is sent through untouched.
        assert json.loads(request.data.decode("utf-8")) == {
            "model": RECIPE_MODEL,
            "input": ["hash password", "naïve façade ✓"],
        }
        # The learned dimension is local metadata only.
        assert provider.dimension == 4


class TestRecipeAgainstLoopbackGateway:
    def test_embed_then_search_round_trip_with_slash_model(
        self, monkeypatch, tmp_path, fake_gateway,
    ):
        port, requests = fake_gateway
        base_url = f"http://127.0.0.1:{port}/v1"
        _apply_recipe(monkeypatch, CRG_OPENAI_BASE_URL=base_url)
        nodes = [
            _node("hash_password", "pkg/mod.py::hash_password", "Hash a pässword."),
            _node("make_invoice", "pkg/mod.py::make_invoice", "Build an invoice."),
            _node("noop", "pkg/mod.py::noop"),
        ]

        with EmbeddingStore(tmp_path / "graph.db", provider="openai") as store:
            assert store.embed_nodes(nodes) == 3
            # Idempotent: unchanged text under the same identity is not resent.
            assert store.embed_nodes(nodes) == 0
            ranked = store.search("password reset", limit=3)
            providers = {
                row[0] for row in store._conn.execute("SELECT provider FROM embeddings")
            }

        assert ranked[0][0] == "pkg/mod.py::hash_password"
        assert providers == {f"openai:{RECIPE_MODEL}@{base_url}"}
        # One batch for the nodes, one for the query; nothing for the re-run.
        assert [r["path"] for r in requests] == ["/v1/embeddings", "/v1/embeddings"]
        assert all(r["authorization"] == f"Bearer {RECIPE_KEY}" for r in requests)
        assert all(r["payload"]["model"] == RECIPE_MODEL for r in requests)
        assert all("dimensions" not in r["payload"] for r in requests)
        assert any("pässword" in text for text in requests[0]["payload"]["input"])

    def test_moving_an_index_between_endpoints_reembeds_and_never_mixes(
        self, monkeypatch, tmp_path, fake_gateway,
    ):
        """Same model id, different endpoint: vectors must not be shared."""
        port, requests = fake_gateway
        first = f"http://127.0.0.1:{port}/v1"
        second = f"http://localhost:{port}/v1"
        nodes = [
            _node("hash_password", "pkg/mod.py::hash_password"),
            _node("make_invoice", "pkg/mod.py::make_invoice"),
        ]
        db = tmp_path / "graph.db"

        _apply_recipe(monkeypatch, CRG_OPENAI_BASE_URL=first)
        with EmbeddingStore(db, provider="openai") as store:
            assert store.embed_nodes(nodes) == 2

        _apply_recipe(monkeypatch, CRG_OPENAI_BASE_URL=second)
        with EmbeddingStore(db, provider="openai") as store:
            # Nothing stored under the new identity yet: search sees nothing.
            assert store.search("password") == []
            assert store.embed_nodes(nodes) == 2
            assert store.embed_nodes(nodes) == 0
            stored = {
                row[0] for row in store._conn.execute("SELECT provider FROM embeddings")
            }
            assert stored == {f"openai:{RECIPE_MODEL}@{second}"}
        # First endpoint's batch, the query, the second endpoint's batch; the
        # idempotent re-run sends nothing.
        assert len(requests) == 3
        assert all(r["payload"]["model"] == RECIPE_MODEL for r in requests)

    def test_rejected_key_surfaces_gateway_message_without_the_key(
        self, monkeypatch, tmp_path, fake_gateway,
    ):
        port, requests = fake_gateway
        wrong_key = "wrong-secret-zzz"
        _apply_recipe(
            monkeypatch,
            key=wrong_key,
            CRG_OPENAI_BASE_URL=f"http://127.0.0.1:{port}/v1",
        )
        provider = get_provider("openai")
        with pytest.raises(RuntimeError) as excinfo:
            provider.embed(["hash password"])

        message = str(excinfo.value)
        assert message == "OpenAI API HTTP 401: invalid api key"
        assert wrong_key not in message
        # A 401 is not retried.
        assert len(requests) == 1
