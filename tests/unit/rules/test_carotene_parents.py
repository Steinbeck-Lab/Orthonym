""" carotene parent retained names (v47 P2 / A2 slice-1).

The 28 fundamental C40 carotene parents (all β,γ,ε,κ,φ,χ,ψ end-group pairs) are
recognized by exact InChIKey (data/natural_products.py::CAROTENE_PARENT_INCHIKEYS)
and named with their Greek retained PIN. Structures were derived from OPSIN 2.9.0;
the end-group citation order β,γ,ε,κ,φ,χ,ψ was verified via OPSIN (it accepts
`gamma,epsilon-carotene`, rejects `epsilon,gamma-carotene`). Stereo-exact: only the
all-E fundamental parent matches, so cis/modified carotenoids fall through to
systematic nomenclature (fail-closed). 0-wrong: the Greek name is OPSIN-RT gated.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.namer import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.validation import opsin_roundtrip_check
from orthonym.data.natural_products import CAROTENE_PARENT_INCHIKEYS

# Representatives across the end-group types (β 6-ring, ε 6-ring, γ exocyclic
# methylene, κ cyclopentane, φ aromatic, ψ acyclic). RAW strings: several canonical
# SMILES carry `\` stereo bonds.
REP = [
    (r"CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)CCCC2(C)C)C(C)(C)CCC1",
     "β,β-carotene"),
    (r"CC1=CCCC(C)(C)C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C1=C(C)CCCC1(C)C",
     "β,ε-carotene"),
    (r"C=C1CCCC(C)(C)C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C1C(C)=CCCC1(C)C",
     "γ,ε-carotene"),
    (r"CC(/C=C/C=C(C)/C=C/CC1(C)CCCC1(C)C)=C\C=C\C=C(C)\C=C\C=C(C)\C=C\CC1(C)CCCC1(C)C",
     "κ,κ-carotene"),
    (r"CC(/C=C/C=C(C)/C=C/c1c(C)ccc(C)c1C)=C\C=C\C=C(C)\C=C\C=C(C)\C=C\c1c(C)ccc(C)c1C",
     "φ,φ-carotene"),
    (r"CC(C)=CCC/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C=C(\C)CCC=C(C)C",
     "ψ,ψ-carotene"),
]


@pytest.mark.roundtrip
@pytest.mark.opsin_gate  # pin_verified needs the round trip (claims conformance R63)
@pytest.mark.parametrize("smiles,expected", REP)
def test_carotene_parent_pin(smiles, expected):
    """A fundamental carotene parent names as its Greek retained PIN at pin_verified
    and full-InChIKey round-trips (0-wrong)."""
    with jvm_slots(1, purpose="v47-p2-carotene"):
        eng = Orthonym(style="pin")
        assert eng.name(smiles) == expected
        r = eng.name_tiered(smiles)
        assert r["tier"] == "pin_verified" and r["is_pin"] is True, r
        rt = opsin_roundtrip_check(smiles, expected)
        assert rt.get("passed"), rt.get("error")
        assert (inchi.MolToInchiKey(Chem.MolFromSmiles(rt["opsin_smiles"]))
                == inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)))


def test_carotene_table_integrity():
    """All 28 parents present; each key is a well-formed InChIKey mapping to an
    `<eg>,<eg>-carotene` name over the seven Greek end groups."""
    assert len(CAROTENE_PARENT_INCHIKEYS) == 28
    egs = set("βγεκφχψ")
    for ik, name in CAROTENE_PARENT_INCHIKEYS.items():
        assert len(ik.split("-")) == 3, ik
        assert name.endswith("-carotene"), name
        a, b = name[: -len("-carotene")].split(",")
        assert a in egs and b in egs, name


def test_cis_carotene_falls_through():
    """A (9Z) isomer must NOT get the bare all-E retained name (stereo-exact,
    fail-closed) — the E/Z-modified name is out of slice-1, so it degrades."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    with jvm_slots(1, purpose="v47-p2-carotene"):
        cis = opsin_parse("(9Z)-β,β-carotene")
        assert cis
        assert Orthonym(style="pin").name(cis) != "β,β-carotene"
