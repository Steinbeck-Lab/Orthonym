"""orthonym.jars: the pinned OPSIN / centres jars are found, fetched and checked,
and a missing jar refuses loudly unless reduced mode is requested."""
import hashlib
import http.server
import importlib.util
import multiprocessing as mp
import threading
from pathlib import Path

import pytest

import orthonym.jars as jars

PAYLOAD = b"not really a jar, but pinned bytes" * 100
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def server():
    """Serve PAYLOAD at /good.jar and different bytes at /bad.jar on localhost."""
    hits = {"n": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            body = PAYLOAD if self.path == "/good.jar" else b"tampered"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", hits
    srv.shutdown()


@pytest.fixture
def fake_spec(monkeypatch, tmp_path, server):
    """Point the 'opsin' spec at the local server and an empty jar dir."""
    base, hits = server
    spec = jars.JarSpec(kind="opsin", version="9.9.9", filename="fake-opsin.jar",
                        url=f"{base}/good.jar", sha256=PAYLOAD_SHA, env_var="ORTHONYM_OPSIN_JAR")
    monkeypatch.setitem(jars.JARS, "opsin", spec)
    for var in ("ORTHONYM_OPSIN_JAR", "ORTHONYM_CENTRES_JAR", "ORTHONYM_NO_DOWNLOAD",
                "ORTHONYM_ALLOW_REDUCED", "ORTHONYM_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(tmp_path / "jars"))
    jars._reset_cache()
    yield spec, hits
    jars._reset_cache()


def test_pinned_specs_are_the_official_release_assets():
    assert jars.JARS["opsin"].url == ("https://github.com/dan2097/opsin/releases/download/2.9.0/"
                                      "opsin-cli-2.9.0-jar-with-dependencies.jar")
    assert jars.JARS["opsin"].sha256 == "c2e29326c281f87b59a05d934d8589adac6e9d17b95b984931b3e739111b360f"
    assert jars.JARS["centres"].url == "https://github.com/SiMolecule/centres/releases/download/1.2.1/centres.jar"
    assert jars.JARS["centres"].sha256 == "6c5d5cafa9393ed7b667a2340222348cd15a9bbdd0d1b92518229f26cc5e87fb"


def test_download_then_cache_hit(fake_spec, tmp_path):
    spec, hits = fake_spec
    p = jars.find_jar("opsin")
    assert Path(p).read_bytes() == PAYLOAD and hits["n"] == 1
    jars._reset_cache()
    assert jars.find_jar("opsin") == p and hits["n"] == 1  # verified file reused, no second download
    assert not [f for f in (tmp_path / "jars").iterdir() if f.name.startswith(".fake-opsin")]  # no temp left


def test_bad_checksum_download_is_refused_and_not_kept(fake_spec, monkeypatch, tmp_path):
    spec, _ = fake_spec
    monkeypatch.setitem(jars.JARS, "opsin", jars.JarSpec(**{**spec.__dict__, "url": spec.url.replace("good", "bad")}))
    with pytest.raises(jars.JarUnavailable, match="SHA-256"):
        jars.find_jar("opsin")
    assert not (tmp_path / "jars" / spec.filename).exists()


def test_existing_file_with_wrong_checksum_is_not_overwritten(fake_spec, tmp_path):
    spec, hits = fake_spec
    (tmp_path / "jars").mkdir()
    (tmp_path / "jars" / spec.filename).write_bytes(b"something else")
    with pytest.raises(jars.JarUnavailable, match="does not match"):
        jars.find_jar("opsin")
    assert (tmp_path / "jars" / spec.filename).read_bytes() == b"something else" and hits["n"] == 0


def test_no_download_refuses_and_reduced_mode_returns_none(fake_spec, monkeypatch):
    _, hits = fake_spec
    monkeypatch.setenv("ORTHONYM_NO_DOWNLOAD", "1")
    with pytest.raises(jars.JarUnavailable, match="fetch-jars"):
        jars.find_jar("opsin")
    monkeypatch.setenv("ORTHONYM_ALLOW_REDUCED", "1")
    assert jars.find_jar("opsin") is None and hits["n"] == 0


def test_env_override(fake_spec, monkeypatch, tmp_path):
    f = tmp_path / "mine.jar"
    f.write_bytes(b"x")
    monkeypatch.setenv("ORTHONYM_OPSIN_JAR", str(f))
    assert jars.find_jar("opsin") == str(f.resolve())
    jars._reset_cache()
    monkeypatch.setenv("ORTHONYM_OPSIN_JAR", str(tmp_path / "missing.jar"))
    with pytest.raises(jars.JarUnavailable, match="not a file"):
        jars.find_jar("opsin")


def test_non_pinned_version_is_none(fake_spec):
    assert jars.find_jar("opsin", "2.8.0") is None


def test_outcome_is_resolved_once_per_process(fake_spec, monkeypatch):
    monkeypatch.setenv("ORTHONYM_NO_DOWNLOAD", "1")
    with pytest.raises(jars.JarUnavailable):
        jars.find_jar("opsin")
    monkeypatch.delenv("ORTHONYM_NO_DOWNLOAD")
    with pytest.raises(jars.JarUnavailable):  # still the cached failure: no mid-run mode switch
        jars.find_jar("opsin")
    jars._reset_cache()
    assert jars.find_jar("opsin")


def _child(q):
    try:
        q.put(jars.find_jar("opsin"))
    except Exception as exc:  # pragma: no cover
        q.put(repr(exc))


def test_concurrent_first_use_download(fake_spec):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_child, args=(q,)) for _ in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    paths = {q.get() for _ in procs}
    assert len(paths) == 1
    assert Path(paths.pop()).read_bytes() == PAYLOAD


def test_fetch_all_ignores_no_download(fake_spec, monkeypatch):
    spec, hits = fake_spec
    monkeypatch.setitem(jars.JARS, "centres", jars.JarSpec(**{**spec.__dict__, "kind": "centres",
                        "filename": "fake-centres.jar", "env_var": "ORTHONYM_CENTRES_JAR"}))
    monkeypatch.setenv("ORTHONYM_NO_DOWNLOAD", "1")
    paths = jars.fetch_all(verbose=False)
    assert set(paths) == {"opsin", "centres"} and all(Path(p).read_bytes() == PAYLOAD for p in paths.values())


def test_module_is_stdlib_only_and_loads_by_path():
    src = Path(jars.__file__).read_text()
    assert "import orthonym" not in src and "from orthonym" not in src and "from ." not in src
    spec = importlib.util.spec_from_file_location("_jars_by_path", jars.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.JARS["opsin"].version == "2.9.0"
