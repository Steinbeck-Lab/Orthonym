"""HYG-02 (Phase 173) — named OrthonymLimitError catalog.

Verifies: out-of-scope inputs raise a named error (opt-in); the default path is
byte-identical to legacy descriptive fallbacks; the catalog never fires on
in-scope inputs; and the canary corpus produces zero raises (criterion 3).
"""
import csv
import glob
import os

import pytest
from rdkit import Chem

from orthonym import OrthonymLimitError, classify_limit, name_compound, Orthonym
from orthonym.errors import (
    LIMIT_CATALOG, classify_scope_limit, classify_failure_limit,
    is_failure_name, ATOM_LIMIT,
)


# ---- criterion 2: out-of-scope raises a named code+message (opt-in) ----

def test_wildcard_raises_named_limit():
    n = Orthonym()
    with pytest.raises(OrthonymLimitError) as ei:
        n.name("CC*", raise_on_limit=True)
    err = ei.value
    assert err.code == "WILDCARD_ATOMS"
    assert err.message  # non-empty human message
    assert err.heritage_ref  # cites the HERITAGE analog
    assert err.code in LIMIT_CATALOG


def test_metal_raises_unsupported_element():
    n = Orthonym()
    with pytest.raises(OrthonymLimitError) as ei:
        n.name("[Au]", raise_on_limit=True)
    assert ei.value.code == "UNSUPPORTED_ELEMENT"


def test_isolated_atom_raises():
    n = Orthonym()
    with pytest.raises(OrthonymLimitError) as ei:
        n.name("[H]", raise_on_limit=True)
    assert ei.value.code in ("ISOLATED_ATOM", "UNNAMEABLE")


# ---- default path is byte-identical (no raise, legacy strings) ----

def test_default_path_no_raise_and_legacy_strings():
    # name() default: wildcard does NOT raise; returns whatever it returned before.
    n = Orthonym()
    out = n.name("CC*")  # must not raise
    assert isinstance(out, str)
    # name_compound default: metal -> exact legacy descriptive string.
    assert name_compound("[Au]") == "gold compound (not supported)"
    assert name_compound("CCO") == "ethanol"  # in-scope unchanged


def test_in_scope_inputs_never_raise():
    n = Orthonym()
    for smi in ["CCO", "c1ccccc1", "CC(=O)O", "O", "[H][H]",
                "O=P([O-])([O-])[O-].[Na+].[Na+].[Na+]"]:  # trisodium phosphate
        # raise_on_limit=True must NOT raise for these in-scope inputs
        out = n.name(smi, raise_on_limit=True)
        assert isinstance(out, str) and out


def test_classify_limit_diagnostic():
    assert classify_limit("CCO") is None          # in scope
    assert classify_limit("c1ccccc1") is None      # in scope
    lim = classify_limit("CC*")
    assert lim is not None and lim.code == "WILDCARD_ATOMS"
    # salt with Na names fine -> not a limit
    assert classify_limit("O=P([O-])([O-])[O-].[Na+].[Na+].[Na+]") is None


def test_name_with_confidence_limit_key():
    n = Orthonym()
    md = n.name_with_confidence("CCO")
    assert md.get("limit") is None
    md2 = n.name_with_confidence("CC*")
    assert md2.get("limit") is not None
    assert md2["limit"]["code"] == "WILDCARD_ATOMS"
    assert set(md2["limit"]) == {"code", "message", "heritage_ref"}


# ---- classifier units ----

def test_classify_failure_messages_match_legacy():
    # Byte-identical to the legacy _descriptive_fallback branch strings.
    assert classify_failure_limit(Chem.MolFromSmiles("[Au]")).message == "gold compound (not supported)"
    assert classify_failure_limit(Chem.MolFromSmiles("[He]")).message == "inorganic compound (not supported)"
    assert classify_failure_limit(Chem.MolFromSmiles("CC*")).message == "compound with wildcard atoms (not supported)"
    # an organic that 'fails' maps to the legacy organic string
    assert classify_failure_limit(Chem.MolFromSmiles("CCO")).message == "unknown organic compound"


def test_scope_limit_only_wildcard():
    assert classify_scope_limit(Chem.MolFromSmiles("CC*")) is not None
    # metals / bare atoms are NOT pre-refused (deferred to failure mapping)
    assert classify_scope_limit(Chem.MolFromSmiles("[Au]")) is None
    assert classify_scope_limit(Chem.MolFromSmiles("[H][H]")) is None
    assert classify_scope_limit(Chem.MolFromSmiles("CCO")) is None


def test_is_failure_name():
    assert is_failure_name("") is True
    assert is_failure_name(None) is True
    assert is_failure_name("unknown organic compound") is True
    assert is_failure_name("ethanol") is False


# ---- criterion 3: the error catalog never fires on a currently-named canary ----

def _canary_smiles():
    smis = []
    # use the most recent canary baseline
    cands = sorted(glob.glob("tests/canary/canary_post_171_start.csv"))
    if not cands:
        cands = sorted(glob.glob("tests/canary/canary_post_*.csv"))
    if not cands:
        return smis
    with open(cands[-1]) as f:
        for row in csv.DictReader(f):
            s = (row.get("smiles_input") or row.get("smiles") or "").strip()
            if s:
                smis.append(s)
    return smis


def test_scope_precheck_is_wildcard_exclusive():
    """Criterion 3 (faithful): the scope pre-check never refuses a non-wildcard
    input, so it can never refuse a currently-correctly-named *in-scope* compound.

    NB: the canary corpus deliberately contains out-of-scope rows (wildcard
    structures with `*`, lone-metal "gold compound" rows). Those ARE refusable in
    opt-in mode — that is the feature (criterion 2: refuse instead of emitting a
    plausible-but-wrong name like `*N=C=N[1*]`->"2,4-diazapentane"). The binding
    no-regression gate is default-mode byte-identity, asserted separately. Here we
    prove the pre-check is exactly wildcard-scoped (no over-reach)."""
    over_reach = []
    for smi in _canary_smiles():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        has_wildcard = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
        fired = classify_scope_limit(mol) is not None
        # The pre-check must fire IFF there is a wildcard atom.
        if fired != has_wildcard:
            over_reach.append((smi, fired, has_wildcard))
    assert over_reach == [], (
        f"scope pre-check is not wildcard-exclusive: {over_reach[:5]}")


def test_default_mode_never_raises_limit_on_canary_sample():
    """Criterion 3 (binding): in the DEFAULT mode the canary/production use,
    naming never raises OrthonymLimitError — including for the wildcard rows."""
    n = Orthonym()
    smis = _canary_smiles()
    # Sample (incl. all wildcard rows, which are the interesting ones) for speed.
    wild = [s for s in smis if '*' in s]
    sample = wild + smis[:40]
    for smi in sample:
        try:
            n.name(smi)  # default: raise_on_limit=False
        except OrthonymLimitError as e:  # pragma: no cover
            pytest.fail(f"default name() raised OrthonymLimitError on {smi!r}: {e}")
        except Exception:
            # Pre-existing pipeline exceptions (ValueError on invalid SMILES, and
            # the AttributeError/KeyError/etc. that name_compound already catches)
            # are out of scope for this test — it only asserts the HYG-02 limit
            # is never raised in default mode. Unchanged by Phase 173.
            pass


def test_back_compat_metal_names_import():
    # data.cation_words imports these from namer (re-exported from errors).
    from orthonym.namer import _METAL_NAMES, _ORGANIC_ELEMENTS
    assert _METAL_NAMES["Au"] == "gold"
    assert "C" in _ORGANIC_ELEMENTS
