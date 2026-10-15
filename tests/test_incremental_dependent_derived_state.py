"""Re-parsing an unchanged importer must not wipe its flows and communities.

PR #1062 made a real content change to a file re-parse every file that
imports it (up to ``_MAX_DEPENDENT_FILES``). Re-parsing replaces the
importer's node rows, which gives them new ids and clears ``community_id``.
The update path that ``code-review-graph update`` and the
``build_or_update_graph`` MCP tool share then re-traces flows and re-detects
communities only for ``result["changed_files"]``, so the importer's flows keep
pointing at node ids that no longer exist and its nodes drop out of every
community. A comment-only edit to a widely imported module would empty the
flows and community membership of all its importers.
"""
import subprocess
from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.tools.build import build_or_update_graph
from code_review_graph.tools.flows_tools import get_flow
from code_review_graph.tools.review import get_affected_flows_func

FILES = {
    'fee.ts': 'export function calculateFee(n: number) { return n + 2; }\n',
    # use.ts imports fee.ts, so it is a dependent of fee.ts. Its own bytes
    # never change in this test.
    'use.ts': (
        "import { calculateFee } from './fee';\n"
        'export function main() { return step1(); }\n'
        'function step1() { return step2(); }\n'
        'function step2() { return helper(); }\n'
        'function helper() { return calculateFee(3); }\n'
    ),
    'other.ts': (
        'export function runOther() { return a1(); }\n'
        'function a1() { return a2(); }\n'
        'function a2() { return 1; }\n'
    ),
}
IMPORTER_FUNCTIONS = {'main', 'step1', 'step2', 'helper'}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ['git', '-c', 'user.email=t@test', '-c', 'user.name=t',
         '-c', 'commit.gpgsign=false', '-C', str(repo), *args],
        check=True, capture_output=True,
    )


def _repo(root: Path) -> Path:
    root.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    for name, text in FILES.items():
        (root / name).write_text(text, encoding='utf-8')
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'init')
    return root


def _flow_step_names(repo: Path) -> set[str]:
    flow = get_flow(flow_name='main', repo_root=str(repo))
    assert flow['status'] == 'ok', flow
    return {step['name'] for step in flow['flow']['steps']}


def _importer_community_ids(repo: Path) -> dict[str, int | None]:
    store = GraphStore(repo / '.code-review-graph' / 'graph.db')
    try:
        rows = store._conn.execute(
            "SELECT name, community_id FROM nodes "
            "WHERE file_path = ? AND kind = 'Function'",
            (str(repo / 'use.ts'),),
        ).fetchall()
    finally:
        store.close()
    return {row['name']: row['community_id'] for row in rows}


@pytest.mark.xfail(
    strict=True,
    reason=(
        'Forced dependent re-parses get new node ids, but update post-processing '
        're-traces flows and communities only for result["changed_files"].'
    ),
)
def test_comment_edit_to_imported_module_keeps_importer_flows_and_communities(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv('CRG_DATA_DIR', raising=False)
    repo = _repo((tmp_path / 'repo').resolve())

    built = build_or_update_graph(full_rebuild=True, repo_root=str(repo))
    assert built['status'] == 'ok'
    assert IMPORTER_FUNCTIONS <= _flow_step_names(repo)
    assert all(cid is not None for cid in _importer_community_ids(repo).values())

    # A comment-only edit to the imported module: use.ts is untouched.
    (repo / 'fee.ts').write_text(
        '// pricing helpers\n' + FILES['fee.ts'], encoding='utf-8',
    )
    _git(repo, 'commit', '-qam', 'comment')

    updated = build_or_update_graph(repo_root=str(repo))
    assert updated['status'] == 'ok'
    assert updated['build_type'] == 'incremental'

    # The flow that runs through use.ts must still list use.ts's functions.
    assert IMPORTER_FUNCTIONS <= _flow_step_names(repo)
    # Reviewing use.ts must still find the flow it belongs to.
    affected = get_affected_flows_func(changed_files=['use.ts'], repo_root=str(repo))
    assert affected['status'] == 'ok'
    assert 'main' in {flow['name'] for flow in affected['affected_flows']}
    # use.ts's functions must still belong to a community.
    communities = _importer_community_ids(repo)
    assert set(communities) == IMPORTER_FUNCTIONS
    assert all(cid is not None for cid in communities.values()), communities
