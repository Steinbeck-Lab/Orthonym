"""Only commits of OPSIN's main branch are built. Offline: GitHub is a fake."""
import json
import urllib.error

import pytest

from orthonym import opsin_build as ob

SHA = "e9f62eb38a6f2519056549cc16385a4257c1d55f"


def base_jar(tmp_path):
    base = tmp_path / "base.jar"
    base.write_bytes(b"base")
    return base


@pytest.fixture
def builds(monkeypatch):
    """``build_jar`` and ``self_test`` replaced; the list holds each commit built."""
    built = []

    def build(commit, base, out):
        built.append(commit)
        out.write_bytes(b"built by the test")
        return out
    monkeypatch.setattr(ob, "build_jar", build)
    monkeypatch.setattr(ob, "self_test", lambda jar: None)
    return built


def compare_answer(monkeypatch, status):
    seen = []

    def get(url, timeout, accept=None, limit=0):
        seen.append(url)
        return json.dumps({"status": status}).encode()
    monkeypatch.setattr(ob, "_get", get)
    return seen


@pytest.mark.parametrize("status", ["behind", "identical"])
def test_a_commit_in_the_history_of_main_is_accepted(monkeypatch, status):
    seen = compare_answer(monkeypatch, status)
    ob.require_on_main(SHA)
    assert seen == [f"https://api.github.com/repos/dan2097/opsin/compare/master...{SHA}?per_page=1"]


@pytest.mark.parametrize("status", ["ahead", "diverged", None, 5])
def test_a_commit_outside_the_history_of_main_is_refused(monkeypatch, status):
    compare_answer(monkeypatch, status)
    with pytest.raises(ob.OpsinBuildError, match="not on OPSIN's main branch"):
        ob.require_on_main(SHA)


@pytest.mark.parametrize("failure", [urllib.error.URLError("offline"), urllib.error.HTTPError("u", 403, "limit", {}, None),
                                     urllib.error.HTTPError("u", 404, "no", {}, None), TimeoutError("slow"),
                                     ob.OpsinBuildError("too big")])
def test_when_github_cannot_say_the_commit_is_refused(monkeypatch, failure):
    def get(*a, **k):
        raise failure
    monkeypatch.setattr(ob, "_get", get)
    with pytest.raises(ob.OpsinBuildError, match="cannot check that OPSIN commit"):
        ob.require_on_main(SHA)


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b"null"])
def test_an_answer_that_is_not_an_object_is_a_refusal(monkeypatch, body):
    monkeypatch.setattr(ob, "_get", lambda *a, **k: body)
    with pytest.raises(ob.OpsinBuildError):
        ob.require_on_main(SHA)


def test_a_commit_that_is_not_on_main_is_not_built(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(ob, "build_jar", lambda *a: built.append(1))
    monkeypatch.setattr(ob, "_get", lambda *a, **k: json.dumps({"status": "ahead"}).encode())
    with pytest.raises(ob.OpsinBuildError, match="not on OPSIN's main branch"):
        ob.ensure_jar(SHA, tmp_path / "jars", base_jar(tmp_path))
    assert built == [] and not [p for p in (tmp_path / "jars" / ob.PRIVATE_DIR).iterdir() if p.name != ".lock"]


def test_main_itself_and_a_built_commit_are_not_checked(tmp_path, builds, monkeypatch):
    monkeypatch.setattr(ob, "require_on_main", lambda *a, **k: pytest.fail("asked whether it is on main"))
    monkeypatch.setattr(ob, "resolve_commit", lambda *a, **k: SHA)
    jar_dir = tmp_path / "jars"
    assert ob.ensure_jar("master", jar_dir, base_jar(tmp_path)).commit == SHA  # the tip of main: built, no check
    assert ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path)).commit == SHA  # now built: used, no check
    assert builds == [SHA]
