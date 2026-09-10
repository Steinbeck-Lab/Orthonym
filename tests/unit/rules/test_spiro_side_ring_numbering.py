"""Task C increment 2: saturated heteromonocyclic spiro SIDE-RING numbering.

``rules.spiro._name_side_ring`` used to number a fully-saturated heterocyclic
side ring (imidazolidine, piperidine, pyrrolidine, 1,3-diazinane,...) with an
independent ring walk that disagreed with the stem name: for imidazolidine it
placed the two ring nitrogens at locants 1 and 4 (the NAME fixes them at 1,3)
and handed the spiro CARBON a locant that landed on a nitrogen (``3'``), so the
assembled ``spiro[chromane-4,3'-imidazolidine]`` was rejected by OPSIN (a spiro
junction cannot sit on an N) and the molecule abstained.

``_number_hetero_side_ring`` now numbers the side ring so (a) the heteroatoms
take their canonical lowest locants consistent with the stem /.4)
and (b) the spiro junction takes the lowest locant among those,
DETERMINISTICALLY (independent of RDKit atom order).

Governing rules: IUPAC 2013 /.4,,.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.rules.spiro import name_mixed_spiro_fused, _number_hetero_side_ring


def _const(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m).split("-")[0] if m else None


# (smiles, expected spiro side-ring locant string in the emitted name)
CASES = [
    # hydantoin spiro chromanone: imidazolidine spiro C must be a carbon (4' or 5')
    ("O=C(CN1C(=O)NC2(CCOc3ccccc32)C1=O)NCCC1=CCCCC1", "imidazolidine"),
    # indane spiro imidazolidine
    ("C[C@H](NC(=O)CN1C(=O)N[C@@]2(CCc3ccccc32)C1=O)c1ccc2c(c1)CCCC2",
     "imidazolidine"),
    # tetrahydroquinoline spiro 1,3-diazinane
    ("Cc1ccc2c(c1)C[C@@]1(CN2C)C(=O)NC(=O)N(CCc2ccc(F)cc2)C1=O",
     "1,3-diazinane"),
]


@pytest.mark.parametrize("smi,side_stem", CASES)
def test_side_ring_spiro_locant_lands_on_carbon(smi, side_stem):
    """The side-ring stem is present and the emitted name is OPSIN-parseable to
    the input constitution (a spiro-on-nitrogen locant is not)."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    m = Chem.MolFromSmiles(smi)
    res = name_mixed_spiro_fused(m)
    assert res is not None
    name = res[0]
    assert side_stem in name, name
    parsed = opsin_parse(name)
    assert parsed is not None, f"OPSIN rejected: {name}"
    core_smi = Chem.MolFragmentToSmiles(m, atomsToUse=sorted(res[1]))
    assert _const(parsed) == _const(core_smi), f"{name} -> {parsed}"


def test_imidazolidine_spiro_locant_is_not_a_nitrogen():
    """Direct regression pin for the OPSIN-fatal ``-3'-imidazolidine`` defect."""
    m = Chem.MolFromSmiles("O=C(CN1C(=O)NC2(CCOc3ccccc32)C1=O)NCCC1=CCCCC1")
    res = name_mixed_spiro_fused(m)
    assert res is not None
    # imidazolidine N are at 1,3; the spiro C must be 4' or 5', never 3'
    assert "-3'-imidazolidine" not in res[0], res[0]


def test_side_ring_numbering_is_deterministic():
    """Same molecule from a re-canonicalised SMILES -> byte-identical name."""
    smi = "Cn1cc([C@@H]2N(C(=O)c3cnccn3)CC[C@]23C(=O)Nc2ccccc23)cn1"
    n1 = name_mixed_spiro_fused(Chem.MolFromSmiles(smi))[0]
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    n2 = name_mixed_spiro_fused(Chem.MolFromSmiles(canon))[0]
    assert n1 == n2 == "spiro[2,3-dihydro-1H-indole-3,3'-pyrrolidine]"


def test_sulfone_side_ring_is_lambda2_skeleton_not_lambda4():
    """A ring S(=O)(=O) side ring: the exocyclic =O are SUBSTITUENTS (composed as
    1,1-dioxo), so the ring S stays a plain divalent thioether -> `1,3-thiazolidine`,
    NOT the mis-valenced `1λ4,3-thiazolidine` the degree-based extraction produced
    (it turned the sulfone S into [SH2]). Governing: / (oxide as oxo)."""
    from orthonym.rules.spiro import (
        _name_side_ring, get_spiro_atoms, _classify_rings_around_spiro_center,
    )
    smi = "O=C1CS(=O)(=O)[C@]2(C(=O)N(Cc3ccc(F)cc3)c3ccccc32)N1c1ccc(F)c(F)c1"
    m = Chem.MolFromSmiles(smi)
    sp = list(get_spiro_atoms(m))[0]
    rings = [list(r) for r in m.GetRingInfo().AtomRings()]
    _fr, sr = _classify_rings_around_spiro_center(m, sp, rings)
    name, _map = _name_side_ring(m, sr[0])
    assert name == "1,3-thiazolidine", name       # not 1λ4,3-thiazolidine


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
def test_sulfone_spiro_indoline_round_trips_end_to_end():
    """The sulfone-thiazolidinone spiro oxindole ships 0-wrong (was abstain)."""
    from orthonym.namer import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smi = "O=C1CS(=O)(=O)[C@]2(C(=O)N(Cc3ccc(F)cc3)c3ccccc32)N1c1ccc(F)c(F)c1"
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)
    name = nm.name(smi)
    assert name and name != "unknown organic compound", name
    assert _const(opsin_parse(name)) == _const(smi), name


def test_number_hetero_side_ring_places_heteroatoms_lowest():
    """Unit: imidazolidine ring numbered N=1,3 and the spiro carbon lowest."""
    # spiro[indane-1,4'-imidazolidine] core
    m = Chem.MolFromSmiles("O=C1NC2(CCc3ccccc32)C(=O)N1")
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    # the imidazolidine ring = the 5-ring with two N
    imid = next(r for r in rings
                if sum(1 for a in r if m.GetAtomWithIdx(a).GetSymbol() == "N") == 2)
    locmap = _number_hetero_side_ring(m, list(imid))
    assert locmap is not None
    n_locs = sorted(locmap[a] for a in imid
                    if m.GetAtomWithIdx(a).GetSymbol() == "N")
    assert n_locs == [1, 3], n_locs
