"""M4 lever L1a: name MASKED-SPIRO systems — a spiro atom that is ALSO a
von-Baeyer bridgehead, so it lies in >=3 SSSR rings and ``get_spiro_atoms``
(which needs EXACTLY 2 SSSR rings) misses it.

Before this lever every such P-24.5 spiro-with-von-Baeyer-component system
VOIDED: ``get_spiro_atoms`` returned [] so every spiro namer bailed, and
``analyze_cage_universal`` built a whole-system cage descriptor that
``audit_von_baeyer_descriptor`` correctly rejected (it is spiro, not a pure
cage). The best-effort FLOOR then abstained (``unknown organic compound``).

Detection rule (root-cause, general): within a connected ring system, the TRUE
masked spiro atom is a ring atom in >=3 SSSR rings whose removal splits the
ring-atom-induced subgraph into EXACTLY 2 connected ring-components (a genuine
spiro cut-vertex). An atom in >=3 SSSR rings whose removal leaves 1 component is
a von-Baeyer BRIDGEHEAD, not spiro, and is excluded.

Each side (spiro atom included) is then named as a full parent — a von-Baeyer
bicyclic via ``analyze_cage_universal``, a single ring via the monocycle namer —
and the P-24.5.1 separable name ``spiro[<sideA>-x,y'-<sideB>]`` assembled.

0-wrong is preserved by the caller's offer full-InChIKey RT gate; the PIN /
default path stays byte-identical (it still abstains on these witnesses).

Governing rule: IUPAC 2013 P-24.5 (spiro systems with von-Baeyer components),
P-24.5.1 (separable spiro name).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi


def _ikey(smi):
    """Full InChIKey (constitution + stereo) of a SMILES."""
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m) if m else None


# (smiles, has_defined_stereo) — all five VOIDED before this lever.
WITNESSES = [
    ("NC1=NCC2(CN3CCC2CC3)N1C1CC1", False),
    ("O=C1CC2(CN3CCC2CC3)ON1c1cccs1", False),
    ("FC(F)(F)c1cnc(NC2=NCC3(CN4CCC3CC4)O2)cn1", False),
    ("O=C1O[C@]2(CN3CCC2CC3)CN1c1csc(-c2ccsc2)c1", True),
    ("O=C1CC2(N1)[C@@H]1C=CC=C[C@@H]2C=C1", True),
]


# --- pure-unit: the masked-spiro detector ---------------------------------

def test_detects_exactly_one_masked_spiro_atom():
    """Each witness has exactly one true masked spiro atom (>=3 SSSR rings,
    removal splits its ring system into 2 components); a plain von-Baeyer
    bridgehead (removal leaves 1 component) is excluded."""
    from orthonym.rules.spiro import find_masked_spiro_atoms
    for smi, _stereo in WITNESSES:
        m = Chem.MolFromSmiles(smi)
        masked = find_masked_spiro_atoms(m)
        assert len(masked) == 1, (smi, masked)


def test_plain_bicyclic_bridgehead_is_not_masked_spiro():
    """A pure von-Baeyer cage (quinuclidine) has bridgeheads in >=3 rings but NO
    spiro cut-vertex, so the detector returns nothing (fail-closed)."""
    from orthonym.rules.spiro import find_masked_spiro_atoms
    m = Chem.MolFromSmiles("C1CN2CCC1CC2")  # quinuclidine, no spiro
    assert find_masked_spiro_atoms(m) == set()


# --- the masked-spiro namer builds the P-24.5.1 separable form ------------

def test_masked_spiro_namer_emits_separable_spiro_name():
    """``_name_masked_spiro`` builds a ``spiro[...]`` name whose CORE covers the
    whole masked-spiro ring system."""
    from orthonym.rules.spiro import _name_masked_spiro
    m = Chem.MolFromSmiles("O=C1CC2(N1)[C@@H]1C=CC=C[C@@H]2C=C1")
    res = _name_masked_spiro(m, allow_vonbaeyer=True)
    assert res is not None
    name, core_atoms, _locants, _subs = res
    assert name.startswith("spiro[") or "spiro[" in name, name
    assert "bicyclo[" in name, name


# --- PIN / default path is untouched: still abstains on every witness -----

@pytest.mark.parametrize("smi,_stereo", WITNESSES)
def test_default_pin_path_still_abstains(smi, _stereo):
    """The default (PIN) namer MUST NOT change: these are best-effort-only
    covering names, so the default path keeps abstaining.

    Run in a FRESH subprocess: a pre-existing ``complex_ring`` handler ships a
    stereo-incomplete covering name for one witness only when OPSIN/CIP state is
    warm (the state-accumulation hazard documented in the contributor guide), which would
    make an in-process assertion order-dependent. A clean process shows the true
    default behaviour, and confirms this masked-spiro lever is floor-only."""
    import subprocess, sys
    code = (
        "from orthonym.namer import Orthonym;"
        f"print(Orthonym().name({smi!r}))"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.stdout.strip() == "unknown organic compound", out.stdout


# --- best-effort FLOOR now converts, 0-wrong (full-InChIKey RT) ------------

@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,_stereo", WITNESSES)
def test_best_effort_converts_masked_spiro_full_inchikey(smi, _stereo):
    """The best-effort floor now EMITS a name that OPSIN round-trips to the
    input's FULL InChIKey (constitution AND stereo)."""
    from orthonym.namer import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_parse
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)
    name = nm.name(smi)
    assert name and name != "unknown organic compound", f"abstained: {smi}"
    parsed = opsin_parse(name)
    assert parsed is not None, f"OPSIN could not parse: {name}"
    assert _ikey(parsed) == _ikey(smi), f"wrong molecule: {name} -> {parsed}"
