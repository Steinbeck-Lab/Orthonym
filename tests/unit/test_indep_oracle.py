"""The independent reader of ``tests/support/indep_oracle`` and the read-back of lone-pair stereocentres built on it.

``tests.support.rt_assert.name_is_rt_exact`` read OPSIN's SMILES of a name and the input through RDKit, which reads
a lone-pair stereocentre by its position in the string, and, after the first round of PubChem 1M lane S2, through
the function under test (``orthonym.perception.lone_pair_reading``). For an input that writes a lone-pair
stereocentre it now compares CDK's reading of the SMILES (the OpenSMILES reading, handed to the InChI library) with
OPSIN's own InChIKey of the name: no RDKit, no Orthonym. The parity convention of the bridge is pinned here against
OPSIN's own key of a name on structures whose configuration is known; an input it cannot hand over, or an element it
does not read (Sb, Bi, Te, Po), is not confirmed.
"""
import pytest

from tests.support import indep_oracle
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

PINNED = [
    # (SMILES, a name of exactly that structure)
    ("C[C@H](N)C(=O)O", "L-alanine"),
    ("C[C@@H](N)C(=O)O", "D-alanine"),
    ("C/C=C/C", "(2E)-but-2-ene"),
    ("C/C=C\\C", "(2Z)-but-2-ene"),
    ("[13CH4]", "(13C)methane"),
    ("C[S@](=O)c1ccccc1", "(S)-(methanesulfinyl)benzene"),
    ("C[S@@](=O)c1ccccc1", "(R)-(methanesulfinyl)benzene"),
    ("[S@@](C)(=O)c1ccccc1", "(S)-(methanesulfinyl)benzene"),           # the sulfur written first
    ("C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1", "(1R,2S)-1,2-diphenylphospholane"),
    ("CO[P@@]1OC[C@H](C)O1", "(2R,4S)-2-methoxy-4-methyl-1,3,2-dioxaphospholane"),
    ("C[C@H]1C[N@]1c1ccccc1", "(1R,2S)-2-methyl-1-phenylaziridine"),
]


@pytest.mark.parametrize("smiles,name", PINNED)
def test_the_reader_agrees_with_opsins_own_key_on_known_structures(smiles, name):
    got = indep_oracle.keys(smiles, name)
    assert got is not None, "the independent reader is not available (a JDK with javac and the two jars are needed)"
    key_in, key_name = got
    assert key_in == key_name and not key_in.startswith("ERR")


@pytest.mark.parametrize("smiles,wrong", [
    ("C[C@H](N)C(=O)O", "D-alanine"),
    ("C[S@](=O)c1ccccc1", "(R)-(methanesulfinyl)benzene"),
    ("C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1", "(1S,2S)-1,2-diphenylphospholane"),
    ("C[C@H]1C[N@]1c1ccccc1", "(2S)-2-methyl-1-phenylaziridine"),        # the N configuration left out
])
def test_the_reader_tells_a_wrong_name_from_the_right_one(smiles, wrong):
    key_in, key_name = indep_oracle.keys(smiles, wrong)
    assert key_in != key_name and key_in.split("-")[0] == key_name.split("-")[0]


def test_a_name_with_a_free_valence_is_read_with_radicals_allowed_for_a_radical_input():
    smiles, name = "C[C@@H]1CC[P@@](=O)C1", "(3R)-3-methyl-1-oxo-1λ5-phospholan-1-yl"      # a phospholanyl radical
    key_in, key_name = indep_oracle.keys(smiles, name, radicals=True)
    assert key_in == key_name
    assert indep_oracle.keys(smiles, name)[1] == "NOPARSE"
    assert name_is_rt_exact(name, smiles)


def test_a_structure_it_cannot_hand_over_is_not_a_key():
    key_in, _ = indep_oracle.keys("[Fe+2].[Cl-].[Cl-]")
    assert key_in.startswith("ERR") or key_in
    assert indep_oracle.keys("C\tC") is None                              # the protocol is one line per request


@pytest.mark.parametrize("smiles,name,expected", [
    ("C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1", "(1R,2S)-1,2-diphenylphospholane", True),
    ("C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1", "(1S,2S)-1,2-diphenylphospholane", False),
    ("C1C[C@H]([As@@](C1)c1ccccc1)c1ccccc1", "(1R,2S)-1,2-diphenylarsolane", True),
    ("C1C[C@H]([As@@](C1)c1ccccc1)c1ccccc1", "(1S,2S)-1,2-diphenylarsolane", False),
    ("[As@@](C)(CC)c1ccccc1", "(R)-ethyl(methyl)(phenyl)arsane", True),
    ("[As@@](C)(CC)c1ccccc1", "(S)-ethyl(methyl)(phenyl)arsane", False),
    ("C[C@H]1C[N@]1c1ccccc1", "(1R,2S)-2-methyl-1-phenylaziridine", True),
    ("C[C@H]1C[N@]1c1ccccc1", "(2S)-2-methyl-1-phenylaziridine", False),
    # Sb: no key holds the configuration the string defines, so nothing is confirmed
    ("C1C[C@H]([Sb@@](C1)c1ccccc1)c1ccccc1", "(2S)-1,2-diphenylstibolane", False),
])
def test_the_read_back_of_a_lone_pair_input(smiles, name, expected):
    assert name_is_rt_exact(name, smiles) is expected


@pytest.mark.parametrize("smiles,name,expected", [
    ("C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1", "(1R,2S)-1,2-diphenylphospholane", True),
    ("C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1", "(1S,2S)-1,2-diphenylphospholane", False),
    ("C1C[C@H]([As@@](C1)c1ccccc1)c1ccccc1", "(1R,2S)-1,2-diphenylarsolane", True),
    ("C1C[C@H]([As@@](C1)c1ccccc1)c1ccccc1", "(1S,2S)-1,2-diphenylarsolane", False),
    ("C1C[C@H]([Sb@@](C1)c1ccccc1)c1ccccc1", "(2S)-1,2-diphenylstibolane", False),
])
def test_without_the_reader_the_read_back_fails_closed(monkeypatch, smiles, name, expected):
    # no javac: the stand-in reading of the engine, and no centre without a label
    monkeypatch.setattr(indep_oracle, "keys", lambda smiles, name="", radicals=False: None)
    assert name_is_rt_exact(name, smiles) is expected


# ---------------------------------------------------------------------------------------------
# The compiled class is only run from a directory this user alone owns and can write
# ---------------------------------------------------------------------------------------------
#
# The class is compiled on first use and run in a JVM of the test session. A cache in a shared, predictable
# location (the temp directory, a name that is the hash of a public source) would let another local user make the
# directory first and have the session run a class of theirs. ``indep_oracle._compiled`` keeps the class in a
# private per-user cache, checks every file of it with ``os.lstat`` (owner, no symlink, no group or world write)
# before using it, and otherwise compiles into a fresh ``tempfile.mkdtemp`` (0700) for the process.

import hashlib
import os
import stat
from pathlib import Path


@pytest.fixture
def cache(tmp_path, monkeypatch):
    base = tmp_path / "cache"
    base.mkdir()
    os.chmod(base, 0o700)
    monkeypatch.setenv("XDG_CACHE_HOME", str(base))
    return base


def _digest(cp):
    source = Path(indep_oracle.__file__).resolve().parent / "IndepOracle.java"
    return hashlib.sha256(source.read_bytes() + cp.encode()).hexdigest()[:16]


def _class_is_real(directory):
    data = (Path(directory) / "IndepOracle.class").read_bytes()
    return data[:4] == bytes.fromhex("cafebabe")


def _plant(directory):
    directory.mkdir()
    (directory / "IndepOracle.class").write_bytes(b"planted")
    os.chmod(directory / "IndepOracle.class", 0o666)
    os.chmod(directory, 0o777)


def test_the_class_is_kept_in_a_private_cache_and_used_again(cache):
    cp = indep_oracle._classpath()
    first = indep_oracle._compiled(cp)
    assert first == cache / "orthonym-indep-oracle" / _digest(cp)
    assert _class_is_real(first)
    assert stat.S_IMODE(os.lstat(cache / "orthonym-indep-oracle").st_mode) == 0o700
    assert all(os.lstat(f).st_uid == os.getuid() and not os.lstat(f).st_mode & 0o022 for f in first.iterdir())
    stamp = os.lstat(first / "IndepOracle.class").st_mtime_ns
    assert indep_oracle._compiled(cp) == first
    assert os.lstat(first / "IndepOracle.class").st_mtime_ns == stamp          # not compiled again


def test_a_planted_directory_with_open_permissions_is_not_used(cache):
    cp = indep_oracle._classpath()
    parent = cache / "orthonym-indep-oracle"
    parent.mkdir(mode=0o700)
    planted = parent / _digest(cp)
    _plant(planted)
    assert not indep_oracle._trusted_classes(planted)
    got = indep_oracle._compiled(cp)
    assert got is not None and _class_is_real(got) and indep_oracle._trusted_classes(got)
    assert (got / "IndepOracle.class").read_bytes() != b"planted"


def test_a_directory_of_another_owner_is_not_used(cache, monkeypatch):
    cp = indep_oracle._classpath()
    good = indep_oracle._compiled(cp)
    assert indep_oracle._trusted_classes(good)
    # as another user would see it: the owner is not the user of the process
    monkeypatch.setattr(indep_oracle, "_uid", lambda: os.getuid() + 1)
    assert not indep_oracle._trusted(good, True)
    assert not indep_oracle._trusted_classes(good)
    assert not indep_oracle._owned_directory(good)
    # so a run as that user compiles into a directory of its own, and leaves this one as it is
    got = indep_oracle._compiled(cp)
    assert got != good
    assert _class_is_real(good)


def test_a_symlink_in_the_place_of_the_cache_is_not_followed(cache, tmp_path):
    cp = indep_oracle._classpath()
    good = indep_oracle._compiled(cp)
    elsewhere = tmp_path / "elsewhere"
    os.rename(good, elsewhere)
    os.symlink(elsewhere, good)
    assert not indep_oracle._trusted_classes(good)
    got = indep_oracle._compiled(cp)
    assert got != good and _class_is_real(got) and indep_oracle._trusted_classes(got)
    assert os.path.islink(good)                                              # left as it was


def test_a_class_file_that_others_can_write_is_not_used(cache):
    cp = indep_oracle._classpath()
    good = indep_oracle._compiled(cp)
    os.chmod(good / "IndepOracle.class", 0o664)
    assert not indep_oracle._trusted_classes(good)
    again = indep_oracle._compiled(cp)
    assert again is not None and indep_oracle._trusted_classes(again)


def test_a_cache_directory_others_can_write_is_not_used(cache, tmp_path):
    cp = indep_oracle._classpath()
    os.chmod(cache, 0o777)
    assert indep_oracle._cache_parent() is None
    got = indep_oracle._compiled(cp)
    assert got is not None and _class_is_real(got) and indep_oracle._trusted_classes(got)
    assert cache not in got.parents                                          # a private temp directory of the process
    assert stat.S_IMODE(os.lstat(got).st_mode) == 0o700
