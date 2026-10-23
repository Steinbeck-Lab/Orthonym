"""``orthonym --opsin``: the command-line way to choose the OPSIN, the same as ORTHONYM_OPSIN."""
import os

import pytest

from orthonym import __version__, cli, jars


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # set then remove: monkeypatch then puts the variable back to its first state after a test
    # whose main has set it
    monkeypatch.setenv("ORTHONYM_OPSIN", "")
    monkeypatch.delenv("ORTHONYM_OPSIN")
    monkeypatch.setattr(jars, "_CACHE", {})


def test_version_is_one_line_by_default(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == f"orthonym {__version__}\n"


def test_version_is_one_line_for_the_release_choice(capsys, monkeypatch):
    monkeypatch.setattr(jars, "active_opsin", lambda download=True: pytest.fail("the release jar is not looked up"))
    assert cli.main(["--version", "--opsin", "2.9.0"]) == 0
    assert capsys.readouterr().out == f"orthonym {__version__}\n"


def test_version_names_the_main_build(capsys, monkeypatch):
    sha = "9f30df5cc22c6f9e7f993a497cb3f65fdf4e5fb8"
    seen = []
    monkeypatch.setattr(jars, "active_opsin", lambda download=True: seen.append(download) or
                        {"label": "main", "commit": sha, "path": "/x.jar"})
    assert cli.main(["--opsin", "main", "--version"]) == 0
    assert capsys.readouterr().out == f"orthonym {__version__}\nOPSIN main 9f30df5cc22c\n"
    assert seen == [False]  # the build is looked for, never made, by --version


def test_version_says_when_the_main_build_is_missing(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("ORTHONYM_NO_DOWNLOAD", "1")
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"orthonym {__version__}" and out[1].startswith("OPSIN main (not built yet")
    assert not (tmp_path / "empty").exists()


def test_a_wrong_value_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--opsin", "3.0", "CCO"])
    assert stop.value.code == 2
    assert "2.9.0, main or main@<commit>" in capsys.readouterr().err


def test_a_malformed_choice_in_the_environment_stops_the_run_even_in_reduced_mode(capsys, monkeypatch):
    monkeypatch.setenv("ORTHONYM_OPSIN", "3.0")
    monkeypatch.setenv("ORTHONYM_ALLOW_REDUCED", "1")
    assert cli.main(["CCO"]) == 2
    err = capsys.readouterr().err
    assert "2.9.0, main or main@<commit>" in err and "to name without the jars" not in err


def test_the_option_sets_the_choice_before_a_jar_is_resolved(monkeypatch):
    seen = []
    monkeypatch.setattr(jars, "fetch_all", lambda verbose=True: seen.append(("fetch_all", os.environ.get("ORTHONYM_OPSIN"))))
    monkeypatch.setattr(jars, "fetch_opsin_choice", lambda verbose=True: seen.append(("fetch_opsin", os.environ.get("ORTHONYM_OPSIN"))))
    assert cli.main(["--opsin", "main@9f30df5cc22c", "--fetch-jars"]) == 0
    assert seen == [("fetch_all", "main@9f30df5cc22c"), ("fetch_opsin", "main@9f30df5cc22c")]


def test_the_option_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    monkeypatch.setattr(jars, "fetch_all", lambda verbose=True: None)
    monkeypatch.setattr(jars, "fetch_opsin_choice", lambda verbose=True: None)
    assert cli.main(["--opsin", "2.9.0", "--fetch-jars"]) == 0
    assert os.environ["ORTHONYM_OPSIN"] == "2.9.0"


def test_a_failed_build_stops_fetch_jars_with_status_1(capsys, monkeypatch):
    def refuse(verbose=True):
        raise jars.JarUnavailable("ORTHONYM_OPSIN=main: javac failed")
    monkeypatch.setattr(jars, "fetch_all", lambda verbose=True: None)
    monkeypatch.setattr(jars, "fetch_opsin_choice", refuse)
    assert cli.main(["--opsin", "main", "--fetch-jars"]) == 1
    assert "javac failed" in capsys.readouterr().err


def test_the_help_lists_the_option(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--opsin {2.9.0,main,main@COMMIT}" in text and "ORTHONYM_OPSIN" in text
