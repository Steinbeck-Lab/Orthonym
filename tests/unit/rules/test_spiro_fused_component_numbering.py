"""Task C increment 3: fused-component numbering in a mixed spiro/fused name.

``rules.spiro._name_fused_component`` falls back to a peripheral-walk locant map
(``_synthesize_fused_locants``) when ``name_fused_heterocycle`` /
``name_ortho_fused_bicyclic`` return a NAME but an EMPTY locant map (the
partial-saturation path). For a HETEROATOM-containing fused component the walk
ignored the stem's fixed heteroatom positions and handed the spiro CARBON a
locant that landed on a ring nitrogen — e.g. ``cyclopenta[d]pyrimidine`` numbers
its pyrimidine N at locant 1, so ``spiro[...-1,3'-piperidine]`` put the spiro
junction on that N and OPSIN rejected it.

The fix uses ``compute_fused_numbering`` (the SAME authority that spelled the
name) for a heteroatom-containing component, so the numbering is consistent with
the stem and the spiro carbon lands on a valid carbon locant; the all-carbon
case keeps the legacy walk (which matches the name-builder's carbocyclic hydro
numbering, whereas compute_fused_numbering can pick a desyncing automorphism).

Governing rules: IUPAC 2013 (fused-ring numbering),.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.rules.spiro import name_mixed_spiro_fused


def _const(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m).split("-")[0] if m else None


# heteroatom-containing fused components that used to put the spiro atom on a
# ring nitrogen (OPSIN-fatal) and now round-trip
HETERO_FIXED = [
    ("COc1cccc(F)c1C(=O)N1CCCC2(CCc3cnc(N(C)C)nc32)C1", "cyclopenta[d]pyrimidine"),
    # the retained name: '(9) quinazoline (PIN)', the Blue Book), not
    # 'benzo[d]pyrimidine'
    ("COc1ccc(C)cc1NC(=O)N1CCC2(CC1)NC(=O)c1cccc(Cl)c1N2", "quinazoline"),
    ("O=C1OC2(CCC(O)(C(=O)Nc3ccn(-c4ccccc4)n3)CC2)c2ncccc21", "furo"),
]

# all-carbon fused components must keep their name-consistent numbering; the
# component is cited as its mancude parent (naphthalene) with its saturation in
# front of the spiro name, the Blue Book)
CARBON_UNCHANGED = [
    ("COC(=O)N(C)[C@H]1CCC2(CCN(CCC(C)(C)C)CC2)C2=C1CCC=C2",
     "-spiro[naphthalene-1,4'-piperidine]"),
    ("CC1(C)CCC[C@]2(C)CC3(CC[C@@H]12)OCOO3",
     "-spiro[naphthalene-2,3'-[1,2,4]trioxolane]"),
]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,stem", HETERO_FIXED + CARBON_UNCHANGED)
def test_fused_component_spiro_locant_round_trips(smi, stem):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    m = Chem.MolFromSmiles(smi)
    res = name_mixed_spiro_fused(m)
    assert res is not None
    name = res[0]
    assert stem in name, name
    parsed = opsin_parse(name)
    assert parsed is not None, f"OPSIN rejected: {name}"
    core = Chem.MolFragmentToSmiles(m, atomsToUse=sorted(res[1]))
    assert _const(parsed) == _const(core), f"{name} -> {parsed}"


def test_cyclopenta_pyrimidine_spiro_not_on_nitrogen():
    """Direct pin: the spiro carbon must not take locant 1 (a pyrimidine N)."""
    m = Chem.MolFromSmiles("COc1cccc(F)c1C(=O)N1CCCC2(CCc3cnc(N(C)C)nc32)C1")
    res = name_mixed_spiro_fused(m)
    assert res is not None
    assert "cyclopenta[d]pyrimidine-1," not in res[0], res[0]


def test_hexahydronaphthalene_keeps_low_carbon_locant():
    """All-carbon regression pin: the name's baked-in hydro prefix and the spiro
    descriptor locant must stay in sync (was desynced to 8 by the wrong path)."""
    m = Chem.MolFromSmiles(
        "COC(=O)N(C)[C@H]1CCC2(CCN(CCC(C)(C)C)CC2)C2=C1CCC=C2")
    res = name_mixed_spiro_fused(m)
    assert res is not None
    # (the Blue Book) with (b) then (e) (:3246): indicated
    # hydrogen at 2, then the hydro prefixes. Was
    # "spiro[1,2,3,4,5,6-hexahydronaphthalene-1,4'-piperidine]".
    assert res[0] == (
        "3,4,5,6-tetrahydro-2H-spiro[naphthalene-1,4'-piperidine]"
    ), res[0]
