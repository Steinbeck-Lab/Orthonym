"""Ring assembly of one mancude parent in two indicated-hydrogen states.

IUPAC (the Blue Book) +: a ring assembly of two *identical* mancude
ring systems is named on the one mancude parent stem, with the indicated hydrogen
cited per component ring as required. Aromatic pyridine and its N-substituted 2H
tautomer are the SAME mancude parent (``pyridine``) differing only in indicated-H
placement, so the single-bond assembly ``C1=CCN(c2ccccn2)C=C1`` is
``2H-1,2'-bipyridine`` -- NOT ``2-(1-azacyclohexa-2,4-dien-1-yl)pyridine`` (the
substituent fallback) nor ``1,2'-bi(1,2-dihydropyridine)`` (a wrong-molecule name
built on the partially-saturated component name).

The detector merges the two rings only when BOTH normalize to the SAME known
mancude monocycle AND each is in a mancude state (``_is_mancude_indicated_h_state``
-- not a hydro derivative). A fully-saturated (piperidine) or partly-saturated
(tetrahydropyridine) partner is a genuinely different parent and must NOT be
mis-detected as an assembly (that is what would emit a wrong-molecule name).

Task 9-B3det (a phase). Cites /.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.rings import get_ring_systems
from orthonym.rules.ring_assemblies import (
    detect_ring_assembly,
    _is_mancude_indicated_h_state,
)
from tests.support.hydro_wiring import needs_wiring
from tests.support.rt_assert import assert_full_rt


class TestMancudeTautomerTarget:
    """The 2H-1,2'-bipyridine target /."""

    @pytest.mark.unit
    def test_detected_as_assembly(self):
        #: pyridine + its N-substituted 2H tautomer = one assembly.
        mol = Chem.MolFromSmiles("C1=CCN(c2ccccn2)C=C1")
        info = detect_ring_assembly(mol, get_ring_systems(mol, include_spiro=False))
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "heterocyclic"
        assert info["double_bond_junction"] is False
        assert info["mancude_tautomer"] is True

    @pytest.mark.unit
    def test_names_2H_bipyridine(self):
        # /: mancude stem 'pyridine', 2H on the tautomer ring.
        assert name_compound("C1=CCN(c2ccccn2)C=C1") == "2H-1,2'-bipyridine"


class TestMustNotMisdetect:
    """Two rings joined by a bond that are NOT one mancude parent + its
    indicated-H tautomer must NOT be merged into an assembly: a hydro
    derivative is a different parent, not another indicated-H state)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,label", [
        # pyridine + PIPERIDINE (fully saturated == perhydro, different parent).
        ("C1CCN(c2ccccn2)CC1", "pyridine+piperidine"),
        # pyridine + 1,2,3,6-tetrahydropyridine (two adjacent CH2, not mancude).
        ("C1CC=CCN1c2ccccn2", "pyridine+tetrahydropyridine"),
        # benzene + pyridine (different mancude parents -> 4-phenylpyridine).
        ("c1ccc(-c2ccncc2)cc1", "benzene+pyridine"),
        ("c1ccc(-c2ccccc2)nc1", "benzene+pyridine (2-phenylpyridine)"),
    ])
    def test_not_detected(self, smiles, label):
        mol = Chem.MolFromSmiles(smiles)
        info = detect_ring_assembly(mol, get_ring_systems(mol, include_spiro=False))
        assert info is None, f"{label} wrongly detected as ring assembly: {info}"

    @pytest.mark.unit
    @needs_wiring
    def test_piperidine_names_as_hydro_assembly(self):
        # Leads R2A: the detector above rightly leaves pyridine + piperidine alone (they are no mancude
        # tautomer pair), and the molecule is the hydro-modified assembly of (the Blue Book
        # "When assemblies of otherwise identical rings contain both mancude and saturated rings, the use of
        # hydro prefixes is preferred";:24159 '1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)'): the junction
        # nitrogen carries no double bond,:15593; '2H-1,2'-bipyridine (PIN)',:15634), so the
        # remaining positions are indicated hydrogen 2H and the hydro prefixes 3,4,5,6.
        # It is not '2-(piperidin-1-yl)pyridine', the substituent name this test used to pin (a non-PIN).
        smiles = "C1CCN(c2ccccn2)CC1"
        name = name_compound(smiles)
        assert name == "3,4,5,6-tetrahydro-2H-1,2'-bipyridine"
        assert_full_rt(name, smiles)

    @pytest.mark.unit
    def test_phenylpyridine_unchanged(self):
        assert name_compound("c1ccc(-c2ccncc2)cc1") == "4-phenylpyridine"


class TestSymmetricAssembliesUnchanged:
    """Signature-identical assemblies keep their established naming (they never
    enter the mancude-tautomer merge branch)."""

    @pytest.mark.unit
    def test_symmetric_bipyridine(self):
        assert name_compound("c1ccnc(-c2ccccn2)c1") == "2,2'-bipyridine"

    @pytest.mark.unit
    def test_biphenyl(self):
        assert name_compound("c1ccc(-c2ccccc2)cc1") == "1,1'-biphenyl"


class TestIndicatedHydrogenLowestLocantDeterminism:
    """ BLOCKER-3: the indicated-hydrogen locant must be the LOWEST possible
    and INDEPENDENT of the SMILES atom order.

     (the Blue Book) orders the numbering criteria "heteroatoms have the lower
    possible locants, then indicated hydrogen atoms..." and (f)
    (the Blue Book) "low locants are assigned to indicated hydrogen atoms". For a
    2H-pyridin-1-yl junction the inter-system bond atom IS the ring nitrogen
    (locant 1 in both walking directions), so the connection-locant set ties and
    the indicated-hydrogen locant is the decisive next criterion: 2 < 6 gives
    ``2H-1,2'-bipyridine`` for EVERY writing (the engine used to emit ``6H-`` on
    half the writings -- a non-PIN spelling AND a determinism defect).
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("writings,expected", [
        # 1,2'-bipyridine -- four writings of the SAME molecule (RDKit-canonical
        # C1=CCN(c2ccccn2)C=C1). All must give 2H-, never 6H-.
        ([
            "C1=CCN(c2ccccn2)C=C1",
            "C1=CC=CN(C1)c1ccccn1",
            "C1C=CC=CN1c1ccccn1",
            "c1ccc(N2CC=CC=C2)nc1",
        ], "2H-1,2'-bipyridine"),
        # 1,3'-bipyridine analogue -- same tie, same rule.
        ([
            "C1=CC=CN(C1)c1cccnc1",
            "C1=CCN(c2cccnc2)C=C1",
            "c1cc(N2CC=CC=C2)cnc1",
        ], "2H-1,3'-bipyridine"),
    ])
    def test_indicated_h_lowest_across_writings(self, writings, expected):
        names = {name_compound(s) for s in writings}
        assert names == {expected}, (
            f"order-dependent / non-PIN indicated H: {names}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("writings,expected", [
        # The sibling 9-B3a rows stay correct AND order-independent. Their
        # attachment carbon (or absent indicated H) fixes the walking direction,
        # so the new iH tiebreak never changes them.
        (["C1=COC(C2C=CC=CO2)C=C1", "O1C=CC=CC1C1OC=CC=C1"],
         "2H,2'H-2,2'-bipyran"),
        (["c1ccn(-n2cccc2)c1", "C1=CC=CN1N1C=CC=C1"],
         "1,1'-bipyrrole"),
    ])
    def test_sibling_rows_order_independent(self, writings, expected):
        names = {name_compound(s) for s in writings}
        assert names == {expected}, f"sibling row drifted: {names}"


class TestMancudeStatePredicate:
    """_is_mancude_indicated_h_state: mancude (indicated-H) vs hydro derivative."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,want", [
        ("c1ccccc1", True),        # benzene: fully mancude
        ("c1ccncc1", True),        # pyridine: fully mancude
        ("C1=CC=CN(C)C1", True),   # N-methyl-2H-pyridine: mancude + indicated H
        ("C1CCCCC1", False),       # cyclohexane: perhydro
        ("C1CCNCC1", False),       # piperidine: perhydro
        ("C1CC=CCN1C", False),     # tetrahydropyridine: two adjacent CH2
    ])
    def test_predicate(self, smiles, want):
        mol = Chem.MolFromSmiles(smiles)
        # the whole molecule is one ring system here
        atoms = set(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
        assert _is_mancude_indicated_h_state(mol, atoms) is want
