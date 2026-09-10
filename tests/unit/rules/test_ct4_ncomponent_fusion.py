""" CT.4 — general N-component fusion (carbocyclic-child naphtho case).

Extends the existing polycomponent constructor's deferred topology:
a senior heterocyclic 2-ring BASE (quinoxaline/quinoline/...) ortho-fused to a
naphthalene 2-ring carbocyclic PREFIX -> naphtho[2,3-g]quinoxaline. This UPGRADES
the RT-true von-Baeyer degradation (valid T3) to the fusion PIN; a case that
cannot be assembled correctly STAYS on the von-Baeyer name (never wrong, never
abstain).

Every emission is OPSIN-round-trip-gated on the FULL InChIKey (0-wrong by
construction). Fresh process per naming probe (feedback_spy_before_you_refute).
"""
import json
import subprocess
import sys

import pytest

# Best-effort tier config (the census / head-to-head config).
_THIS = __file__


def _name_one(smiles):
    """Name ONE smiles in a FRESH subprocess at best-effort tier; return
    {name, tier, rt} where rt is the full-InChIKey OPSIN round-trip verdict."""
    out = subprocess.run(
        [sys.executable, _THIS, smiles],
        capture_output=True, text=True,
    )
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    assert line, f"no JSON from subprocess: {out.stdout!r} / {out.stderr[-800:]!r}"
    return json.loads(line[-1])


# --- the two upgrade witnesses: von-Baeyer -> fusion PIN --------------------
def test_naphtho_quinoxaline_exact_pin():
    """4-component carbocyclic-child fusion must build exactly the OPSIN-feasible
    PIN ``naphtho[2,3-g]quinoxaline`` (was: RT-true von-Baeyer 'tetracyclo[...]')."""
    res = _name_one("c1ccc2cc3cc4nccnc4cc3cc2c1")
    assert res.get("name") == "naphtho[2,3-g]quinoxaline", (
        f"expected naphtho[2,3-g]quinoxaline, got {res.get('name')!r}")
    assert res.get("rt") is True


def test_naphtho_quinoline_exact_pin():
    """4-component carbocyclic-child fusion must build exactly
    ``naphtho[2,3-g]quinoline``."""
    res = _name_one("c1ccc2cc3cc4ncccc4cc3cc2c1")
    assert res.get("name") == "naphtho[2,3-g]quinoline", (
        f"expected naphtho[2,3-g]quinoline, got {res.get('name')!r}")
    assert res.get("rt") is True


# --- broader naphtho class (angular + assorted base letters) ----------------
_NAPHTHO_CLASS = {
    "c1cnc2cc3c(ccc4ccccc43)cc2c1": "naphtho[2,1-g]quinoline",
    "c1cnc2cc3ccc4ccccc4c3cc2c1":   "naphtho[1,2-g]quinoline",
    "c1ccc2cc3nc4ccccc4nc3cc2c1":   "naphtho[2,3-b]quinoxaline",
    "c1ccc2cc3cc4ncncc4cc3cc2c1":   "naphtho[2,3-g]quinazoline",
    "c1ccc2cc3nc4ccccc4cc3cc2c1":   "naphtho[2,3-b]quinoline",
    "c1ccc2cc3cc4cnncc4cc3cc2c1":   "naphtho[2,3-g]phthalazine",
}


@pytest.mark.parametrize("smiles,expected", list(_NAPHTHO_CLASS.items()))
def test_naphtho_class_exact_pin(smiles, expected):
    res = _name_one(smiles)
    assert res.get("name") == expected, (
        f"expected {expected}, got {res.get('name')!r} for {smiles}")
    assert res.get("rt") is True


# --- control: 3-component fusion PIN unchanged (byte-identical) -------------
def test_pyrimido_quinoline_control_byte_identical():
    res = _name_one("c1ccc2nc3ncncc3cc2c1")
    assert res.get("name") == "pyrimido[4,5-b]quinoline", (
        f"3-component control regressed: {res.get('name')!r}")
    assert res.get("rt") is True


# --- fail-closed: cases OUTSIDE the built class keep von-Baeyer / abstain ----
def test_het_het_bicyclic_prefix_stays_zero_wrong():
    """het+het 4-component (quinoxaline+quinoline attached) is a NAMED follow-on
    (heterocyclic bicyclic prefix, not naphtho): it must NOT ship a wrong
    molecule -- it stays on the RT-true von-Baeyer name."""
    res = _name_one("c1ccc2nc3ccc4ncccc4c3nc2c1")
    name = res.get("name") or ""
    if name and "unknown" not in name:
        assert res.get("rt") is True, (
            f"het+het 4-component emitted a non-round-tripping name: {name!r}")


def test_all_carbon_tetracene_not_misnamed_as_fusion():
    """A charged 4-fused aromatic residual / all-carbon PAH must never be
    force-named a naphtho fusion PIN (0-wrong): it round-trips or abstains."""
    # naphthacene (tetracene) -- all-carbon 4-ring, no ring heteroatom.
    res = _name_one("c1ccc2cc3cc4ccccc4cc3cc2c1")
    name = res.get("name") or ""
    if name and "unknown" not in name:
        assert res.get("rt") is True, (
            f"all-carbon tetracene emitted a non-round-tripping name: {name!r}")
    # and it must not be a naphtho[..] fusion (there is no heterocyclic base)
    assert not name.startswith("naphtho["), (
        f"all-carbon PAH wrongly named as naphtho fusion: {name!r}")


# ---------------------------------------------------------------------------
# subprocess entrypoint (fresh process per witness)
# ---------------------------------------------------------------------------
def _run(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v37-ct4-test"):
        from orthonym.namer import Orthonym
        from orthonym.validation import opsin_roundtrip_check
        namer = Orthonym(
            style="pin", general_fallback=True,
            general_fallback_unverified=True, allow_aromatic_general=True)
        res = namer.name_tiered(smiles)
        name = res.get("name")
        rt = None
        if name and "unknown" not in name:
            rt = bool(opsin_roundtrip_check(smiles, name)["passed"])
        return {"name": name, "tier": res.get("tier"), "rt": rt}


if __name__ == "__main__":
    print(json.dumps(_run(sys.argv[1])))
