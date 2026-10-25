"""The von Baeyer cage ceilings by tier.

``vonbaeyer_universal`` analyses a ring system only up to ``MAX_CAGE_ATOMS`` = 40
skeletal atoms and ``MAX_CAGE_RINGS`` = 8 rings at the PIN tier: the
main-bridge selection there is not yet an exhaustive search, so a larger system
could get a correct but non-preferred name. At the best-effort tier a name need
only be a verified systematic name, so the tier takes its measured ceilings
(``BEST_EFFORT_MAX_CAGE_ATOMS`` / ``BEST_EFFORT_MAX_CAGE_RINGS``; TRIAGE 'Large
molecules -- part 3'). The tier is the tier of the outermost naming request
(``provenance.best_effort_request_ctx``), so a best-effort recovery that a PIN
request runs keeps the PIN ceilings, and a name built on a ring system beyond the
PIN ceilings is recorded as not a PIN.

Every expected name is read back by a fresh OPSIN call that does not go through
the engine (``tests.support.rt_assert``).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.jvm_budget import jvm_slots
from orthonym.metrics.provenance import best_effort_ctx, best_effort_request_ctx
from orthonym.rules import vonbaeyer_universal as vbu
from tests.support.opsin_marks import release_opsin_only
from tests.support.rt_assert import name_is_rt_exact

#: calix[6]arene: one ring system of 42 skeletal atoms and 7 rings.
CALIX6 = ("Oc1c2cccc1Cc1cccc(c1O)Cc1cccc(c1O)Cc1cccc(c1O)Cc1cccc(c1O)"
          "Cc1cccc(c1O)C2")
CALIX6_NAME = (
    "heptacyclo[31.3.1.1^3,7.1^9,13.1^15,19.1^21,25.1^27,31]dotetraconta-"
    "1(37),3(42),4,6,9(41),10,12,15(40),16,18,21,23,25(39),27,29,31(38),33,35-"
    "octadecaene-37,38,39,40,41,42-hexol")
#: a cyclic thiazole peptide (ChEBI:107473): a 45-atom ring system, named on the
#: PIN path of the best-effort tier.
CYCLIC_PEPTIDE = (
    "CC[C@H](C)[C@@H]1NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)"
    "[C@@H](CO)NC(=O)CNC(=O)CNC(=O)c2csc(n2)[C@@H](CC(N)=O)NC(=O)[C@H]"
    "(Cc2ccc(O)cc2)NC(=O)c2csc(n2)[C@H](CC(C)C)NC(=O)c2csc1n2")


#: C44H24, one ring system of 44 skeletal atoms and 12 rings (PubChem CID 10674268): the
#: smallest ring count the best-effort ceiling used to refuse (it was 11).
DODECACYCLE = ("C1=CC=C2C(=C1)C=C3C=CC=C4C3=C2C56C4(C7=CC=CC8=C7C5=CC9=CC=CC=C89)C1=CC=CC2="
               "C1C6=CC1=CC=CC=C21")
DODECACYCLE_NAME = (
    "dodecacyclo[13.13.13.1^2,10.1^16,20.1^29,37.0^1,15.0^4,9.0^21,26.0^30,35.0^14,44."
    "0^28,43.0^41,42]tetratetraconta-2,4,6,8,10,12,14(44),16(43),17,19,21,23,25,27,29(42),"
    "30,32,34,36,38,40-henicosaene")
#: a cage of 16 skeletal atoms and 14 rings, here the substituent of a coumarin imine
#: (PubChem CID 59073134).
TETRADECACYCLE = ("C1C2C13C24C35C46C57C68C79C81C92C11C22C11C2C1C=NC1=CC2=C(C=C1)C=CC(=O)O2")
TETRADECACYCLE_PARENT = "tetradecacyclo[7.7.0.0^1,10.0^2,8.0^2,9.0^3,7.0^3,8.0^4,6.0^4,7.0^10,16.0^11,15.0^11,16.0^12,14.0^12,15]hexadecan"
#: a cage of 25 skeletal atoms and 16 rings (PubChem CID 60022746): one ring above the ceiling.
HEXADECACYCLE = "C1[C@@H]2C3[C@@H]4CC5C46C37C28[C@@H]1C9C81C72C63C5C4C33C22C11[C@@H]9CC1C2C3C4"


#: six ring systems (two carbazoles, two dibenzofurans, two phenylenes) joined by bonds: 12 rings
#: as one ring-atom set of 57 atoms, no ring system of more than three rings (PubChem CID 163738885).
#: Named from its parts at the ceiling of 11, and still.
UNION_OF_SYSTEMS = (
    "CC1CC2=C(C=C1)C3=CC=CC=C3N2C4=CC=CC(=C4)C5=CC6=C(C=C5)OC7=C6C=C(C=C7)C8=CC=CC(=C8)C9=CC1=C(C=C9)"
    "OC2=C1C=C(C=C2)C1=CC(=CC=C1)N1C2=CC=CC=C2C2=CC=CC=C21")
UNION_OF_SYSTEMS_NAME = (
    "9-{3-[8-(3-{8-[3-(2-methyl-1,2-dihydro-9H-carbazol-9-yl)phenyl]dibenzo[b,d]furan-2-yl}phenyl)"
    "dibenzo[b,d]furan-2-yl]phenyl}-9H-carbazole")
#: one ring system of 45 atoms and 12 rings that holds a spiro union of two polycyclic parts
#: (PubChem CID 162989189), named from its parts.
SPIRO_UNION = (
    "C[C@H]1CC2=C3[C@H]4[C@H]1C=C5CC[C@@H](C[C@@H]6[C@]7(C8=CC=CC(=C8C(=O)O7)CCCN)[C@]9\\1[C@]6"
    "(C5=C4CC9)C(=O)O/C1=C(\\C[C@H](N1C[C@H]4C[C@@H](C1)[C@H](N3C4)CC2)CC1CCCCC1)/O)C1CCCCC1")


def _row(smiles, tier):
    with jvm_slots(1, purpose="cage-caps-test"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.unit
def test_pin_tier_ceilings_are_unchanged():
    assert (vbu.MAX_CAGE_ATOMS, vbu.MAX_CAGE_RINGS) == (40, 8)
    assert vbu.cage_caps() == (40, 8)


def _in_request(request_tier, fn, *args):
    token = best_effort_request_ctx.set(request_tier)
    try:
        return fn(*args)
    finally:
        best_effort_request_ctx.reset(token)


@pytest.mark.unit
def test_best_effort_ceilings_follow_the_request_tier():
    raised = (vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS)
    assert raised[0] > vbu.MAX_CAGE_ATOMS and raised[1] > vbu.MAX_CAGE_RINGS
    assert _in_request(True, vbu.cage_caps) == raised
    assert _in_request(False, vbu.cage_caps) == (40, 8)
    # the best-effort breadth context alone is no best-effort request: a
    # recovery's fresh best-effort engine sets it inside a PIN request
    token = best_effort_ctx.set(True)
    try:
        assert _in_request(False, vbu.cage_caps) == (40, 8)
        assert vbu.cage_caps() == (40, 8)
    finally:
        best_effort_ctx.reset(token)
    assert vbu.cage_caps() == (40, 8)


@pytest.mark.unit
def test_analyzer_refuses_the_42_atom_system_at_pin_and_analyses_it_at_best_effort():
    from orthonym.metrics.provenance import clear_provenance, get_provenance
    mol = Chem.MolFromSmiles(CALIX6)
    assert vbu.analyze_cage_universal(mol, allow_mancude=True) is None
    def analyse():
        return vbu.analyze_cage_universal(mol, allow_mancude=True)

    assert _in_request(False, analyse) is None
    clear_provenance()
    cage = _in_request(True, analyse)
    assert cage is not None and len(cage.cage_atoms) == 42
    assert cage.descriptor == "heptacyclo[31.3.1.1^3,7.1^9,13.1^15,19.1^21,25.1^27,31]"
    # a system beyond the PIN ceilings is recorded as a part that is never a PIN
    assert cage.descriptor in (get_provenance().get("non_pin_fragments") or ())


@pytest.mark.opsin_gate
def test_a_best_effort_recovery_inside_a_pin_request_keeps_the_pin_ceilings():
    # The fragment rescue builds a fresh best-effort engine inside a request of
    # any tier (decomposition.engine._name_fragment_t4_rescue_fresh). In a PIN
    # request it analyses no ring system beyond the PIN ceilings.
    from orthonym.decomposition.engine import _name_fragment_t4_rescue_fresh
    with jvm_slots(1, purpose="cage-caps-test"):
        assert _in_request(True, _name_fragment_t4_rescue_fresh, CALIX6) == CALIX6_NAME
        assert _in_request(False, _name_fragment_t4_rescue_fresh, CALIX6) is None
        assert _name_fragment_t4_rescue_fresh(CALIX6) is None


@release_opsin_only
@pytest.mark.opsin_gate
def test_best_effort_names_the_calixarene_and_the_pin_tier_refuses():
    row = _row(CALIX6, "best-effort")
    assert row.get("name") == CALIX6_NAME, row.get("name")
    assert name_is_rt_exact(CALIX6_NAME, CALIX6)
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row["tier"]
    pin = _row(CALIX6, "pin")
    assert pin["tier"] == "abstain" and is_failure_name(pin.get("name") or "unknown"), (
        pin.get("name"), pin["tier"])


@pytest.mark.opsin_gate
def test_a_pin_path_name_on_a_raised_cage_is_not_labelled_a_pin():
    row = _row(CYCLIC_PEPTIDE, "best-effort")
    name = row.get("name")
    # superscripts as low as possible, the Blue Book; quick-wins F-Q1),
    # was '...1^23,26.1^33,36...'
    assert name and "tetracyclo[38.2.1.1^5,8.1^15,18]pentatetraconta" in name, name
    assert name_is_rt_exact(name, CYCLIC_PEPTIDE)
    assert row["tier"] != "pin_verified" and row["is_pin"] is False, row["tier"]
    pin = _row(CYCLIC_PEPTIDE, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])


@pytest.mark.unit
def test_name_with_confidence_sets_the_request_tier_for_its_call():
    # name_with_confidence names through name, whose _budget_scope sets the request
    # tier for the outermost call: the ceilings inside follow the engine's tier,
    # and nothing is left behind after the call.
    seen = []

    def fake_impl(self, smiles):
        seen.append((best_effort_request_ctx.get(), vbu.cage_caps()))
        return "methane"

    raised = (vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Orthonym, "_name_impl", fake_impl)
        # the tier label of a wider tier's confidence record runs the strict path again (a call of its own, at the
        # PIN ceilings); the request tier of the call is the subject here
        import orthonym.namer as namer_module
        mp.setattr(namer_module, "tier_label_of", lambda namer, name, smiles: None)
        Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_with_confidence("C")
        Orthonym().name_with_confidence("C")
        # inside a request (an enclosing name scope) the call keeps that
        # request's tier
        from orthonym.assembly.fragment_naming import enter_name_scope, exit_name_scope
        enter_name_scope()
        try:
            _in_request(False, Orthonym(style="pin", **_emit_tier_flags(
                "best-effort")).name_with_confidence, "C")
        finally:
            exit_name_scope()
    assert seen == [(True, raised), (False, (40, 8)), (False, (40, 8))], seen
    assert best_effort_request_ctx.get() is None


#: (41-methyl-1-oxacyclohentetracontan-2-yl)acetic acid: a 41-membered ring,
#: beyond the PIN ceiling of 40 atoms, named as a decorated ring substituent.
OXACYCLE41 = "OC(=O)CC1" + "C" * 38 + "C(C)O1"


@pytest.mark.opsin_gate
def test_a_decorated_ring_substituent_beyond_the_pin_ceilings_is_not_labelled_a_pin():
    from orthonym.metrics.provenance import get_provenance
    row = _row(OXACYCLE41, "best-effort")
    name = row.get("name")
    assert name == "(41-methyl-1-oxacyclohentetracontan-2-yl)acetic acid", name
    assert name_is_rt_exact(name, OXACYCLE41)
    # recorded as not a PIN, as the bare ring substituent is
    assert "1-oxacyclohentetracontan-2-yl" in (get_provenance().get("non_pin_fragments") or ())
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row["tier"]


@pytest.mark.unit
def test_the_best_effort_ring_ceiling_is_the_measured_fifteen():
    # 12-15 rings: the 477 PubChem rows with a ring system of 12 or more rings name
    # identically at every ceiling from 15 to 100, none of them above 15 rings
    # (the sweep table above ``BEST_EFFORT_MAX_CAGE_RINGS``)
    assert vbu.BEST_EFFORT_MAX_CAGE_RINGS == 15
    assert vbu.BEST_EFFORT_MAX_CAGE_ATOMS == 100
    assert _in_request(True, vbu.cage_caps) == (100, 15)
    assert _in_request(False, vbu.cage_caps) == (40, 8)


@pytest.mark.unit
def test_analyzer_admits_a_twelve_and_a_fourteen_ring_cage_at_best_effort_only(monkeypatch):
    from orthonym.metrics.provenance import clear_provenance, get_provenance
    for smiles, ring_word, atoms in ((DODECACYCLE, "dodecacyclo[", 44),
                                     (TETRADECACYCLE, "tetradecacyclo[", 16)):
        mol = Chem.MolFromSmiles(smiles)

        def analyse():
            return vbu.analyze_cage_universal(mol, allow_mancude=True)

        assert _in_request(False, analyse) is None
        clear_provenance()
        cage = _in_request(True, analyse)
        assert cage is not None and len(cage.cage_atoms) == atoms
        assert cage.descriptor.startswith(ring_word), cage.descriptor
        # beyond the PIN ceilings: recorded as a part that is never a PIN
        assert cage.descriptor in (get_provenance().get("non_pin_fragments") or ())
    # the ceiling counts rings the way (circuit rank) and cuts exactly there
    mol = Chem.MolFromSmiles(TETRADECACYCLE)
    monkeypatch.setattr(vbu, "BEST_EFFORT_MAX_CAGE_RINGS", 13)
    assert _in_request(True, lambda: vbu.analyze_cage_universal(mol, allow_mancude=True)) is None
    monkeypatch.setattr(vbu, "BEST_EFFORT_MAX_CAGE_RINGS", 14)
    assert _in_request(True, lambda: vbu.analyze_cage_universal(mol, allow_mancude=True)) is not None


@pytest.mark.unit
def test_analyzer_refuses_a_sixteen_ring_cage_at_best_effort():
    mol = Chem.MolFromSmiles(HEXADECACYCLE)
    assert _in_request(True, lambda: vbu.analyze_cage_universal(mol, allow_mancude=True)) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, parent", [(DODECACYCLE, DODECACYCLE_NAME),
                                            (TETRADECACYCLE, TETRADECACYCLE_PARENT)])
def test_best_effort_names_a_twelve_and_a_fourteen_ring_system_and_the_pin_tier_declines(
        smiles, parent):
    row = _row(smiles, "best-effort")
    name = row.get("name")
    assert name and parent in name, name
    assert name_is_rt_exact(name, smiles)
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row["tier"]
    pin = _row(smiles, "pin")
    assert pin["tier"] == "abstain" and is_failure_name(pin.get("name") or "unknown"), (
        pin.get("name"), pin["tier"])


@pytest.mark.unit
def test_biconnected_ring_system_is_one_system_without_a_spiro_union():
    def atoms_of_rings(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol, {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}

    for smiles, expected in (
            ("c1ccc2ccccc2c1", True),                 # fused
            ("C12C3C4C1C5C2C3C45", True),             # cubane
            ("C1CC2CCC1C2", True),                    # bridged
            ("C1CC1", True),
            ("C1CCC2(CC1)CCCCC2", False),             # spiro union (spiro[5.5]undecane)
            ("C1CCC2(CC1)CC1CCC2CC1", False),         # a spiro union of a monocycle and a bicycle
            ("c1ccccc1-c1ccccc1", False)):            # two ring systems
        mol, ring_atoms = atoms_of_rings(smiles)
        assert vbu.is_biconnected_ring_system(mol, ring_atoms) is expected, smiles
    assert vbu.is_biconnected_ring_system(Chem.MolFromSmiles("C"), set()) is False


@pytest.mark.unit
def test_within_ring_ceiling_keeps_eleven_for_a_set_that_is_not_one_biconnected_system():
    mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCCC2")
    atoms = {a.GetIdx() for a in mol.GetAtoms()}
    union = vbu.BEST_EFFORT_MAX_UNION_RINGS
    assert union == 11 and union < vbu.BEST_EFFORT_MAX_CAGE_RINGS
    # the count alone decides up to 11 rings, and at the PIN tier (ceiling 8)
    assert vbu.within_ring_ceiling(mol, atoms, 11, 15) and vbu.within_ring_ceiling(mol, atoms, 8, 8)
    assert not vbu.within_ring_ceiling(mol, atoms, 9, 8)
    # above 11 only one biconnected system passes: a spiro union does not
    assert not vbu.within_ring_ceiling(mol, atoms, 12, 15)
    cage = Chem.MolFromSmiles("C12C3C4C1C5C2C3C45")
    cage_atoms = {a.GetIdx() for a in cage.GetAtoms()}
    assert vbu.within_ring_ceiling(cage, cage_atoms, 12, 15)
    assert not vbu.within_ring_ceiling(cage, cage_atoms, 16, 15)


@pytest.mark.unit
@pytest.mark.parametrize("smiles, whole_ring_atom_set", [(UNION_OF_SYSTEMS, True), (SPIRO_UNION, False)])
def test_a_union_of_ring_systems_and_a_spiro_union_never_reach_the_main_ring_search(
        smiles, whole_ring_atom_set, monkeypatch):
    # Their circuit rank is 12 or more, above the old ceiling of 11 and within 15: the whole-system
    # analysis used to be refused by the count, and at 15 it ran the main-ring search first and
    # ended in PerfBudgetExceeded, so the parts were never named (21 panel rows lost their names).
    from orthonym.rules.polycyclic import VonBaeyerAnalyzer

    class Reached(BaseException):
        """not an Exception: the analyzer's own ``except Exception`` must not swallow it"""

    def never(*args, **kwargs):
        raise Reached("the von Baeyer analysis ran on a set that is not one ring system")

    monkeypatch.setattr(VonBaeyerAnalyzer, "analyze", never)
    mol = Chem.MolFromSmiles(smiles)
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}

    # the union of ring systems is what the substituent route passes; the spiro union is the largest
    # connected ring component of its molecule (the default of the analyzer)
    cage_atoms = ring_atoms if whole_ring_atom_set else None
    assert _in_request(True, lambda: vbu.analyze_cage_universal(
        mol, cage_atoms=cage_atoms, allow_mancude=True)) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name_part", [(UNION_OF_SYSTEMS, UNION_OF_SYSTEMS_NAME),
                                               (SPIRO_UNION, "spiro[1,3-dihydro-2-benzofuran-1,31'-")])
def test_best_effort_still_names_the_parts_of_a_union_the_ceiling_of_eleven_did(
        smiles, name_part):
    row = _row(smiles, "best-effort")
    name = row.get("name")
    assert name and name_part in name, name
    assert name_is_rt_exact(name, smiles)
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row["tier"]
