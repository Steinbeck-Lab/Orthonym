""" — deep component-numbering: spiro-priority-atom criterion
for tricyclic+ von-Baeyer spiro cages.

PART 2 of the SP2 PIN-spelling pair. Built for PIN-spelling correctness only:
the SP2.0 gate (`V37-a trace-SP2.md` addendum) REFUTED the breadth premise (0/38 of
the spiro-VB abstainer bucket is blocked by deep VB numbering) but CONFIRMED the
structural gap — the tricyclo+ spiro path
(`_tricyclo_plus_spiro_component` -> `analyze_cage_universal` ->
`VonBaeyerAnalyzer.analyze` -> `_choose_lowest_locant_numbering` /
`_locant_criteria_key`) had NO spiro-priority-atom criterion, so the spiro
junction lands at the heteroatom-optimal locant instead of the lowest.

SUCCESS BAR: PIN-spelling correctness + no regression (NOT new emits).

 (a project rule) re-VERIFIED on HEAD in fresh processes:
  * Witness 1's tricyclo cage `3-oxatricyclo[8.3.0.0^2,6]tridecane` numbers the
    spiro junction (orig atom 23) at locant 13 (the MAX in a 13-membered cage);
    the O sits at locant 3.
  * The same-descriptor candidate set offers spiro locants {3, 13}: the
    (spiro=3, O=13) candidate exists but the pre-fix criterion (heteroatom-set
    FIRST) picks (spiro=13, O=3).
 (the Blue Book:10289, PIN `2',12'-dioxaspiro[bicyclo[2.2.1]heptane-2,1'-
cyclododecane]`, "the spiro atom... is given preference for low locant") and
:10186 ("low locants are given to the spiro atom, THEN to the heteroatoms")
mandate the (spiro=3, O=13) numbering -> `13-oxatricyclo[8.3.0.0^2,6]tridecane`.

Both numberings describe the same molecule (renumbering only), so this is a pure
PIN-spelling correction, 0 breadth, RT-invariant.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.spiro import _name_spiro_component
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

# The only true tricyclo+ spiro witness (SP2.0). It abstains at the whole-molecule
# level (a dropped acyloxy substituent, out of SP2 scope); the component-numbering
# is exercised directly below.
W1 = "C=C1C(=O)O[C@H]2[C@H]1[C@@H](OC(=O)[C@](C)(O)CCl)CC(=C)[C@@H]1C[C@H](O)[C@@]3(CO3)[C@H]21"
W1_CAGE = {26, 5, 4, 2, 1, 6, 7, 16, 17, 19, 20, 21, 23}  # 13-membered tricyclo cage
W1_SPIRO = 23


class TestWholeMoleculeUnaffectedControl:
    """The new spiro-priority criterion must be a NO-OP when no spiro atom is
    passed: a plain non-spiro tricyclo+ von-Baeyer PIN stays byte-identical."""

    def test_whole_molecule_vonbaeyer_unaffected_control(self):
        # The control is unaffected by the SPIRO criterion (no spiro atom is passed). Its
        # descriptor moved for a different, deliberate reason: e3b71b4a9 (2026-09-01,
        # " largest main bridge is the PIN") made the main-bridge selection take the
        # LARGEST bridge, where it took the shortest, so this cage's main bicycle is now
        # [15.3.2] (main bridge 2) instead of [9.9.1] (main bridge 1). "Selection of
        # the main ring" (the Blue Book) fixes the 20-atom main ring and
        # "Selection of the main bridge" (:9603; paragraph:9605) takes "the bridge that
        # includes as many of the atoms as possible that are not included in the main ring";
        # symmetric division of the main ring,:9661) applies only after that.
        # Independent enumeration (networkx, not the code under test): the largest simple cycle
        # of this 25-atom graph has 20 atoms, there is exactly one, and the longest bridge of
        # atoms outside it has 2 atoms. Both descriptors read back to the input's full
        # InChIKey with OPSIN 2.9.0 (the structure is the same, so that does not choose between
        # them; the rules above do).
        assert name_compound(
            "C1CC2CCCC3C(C2)C(C1)C1CC3C2CCCC3CCCC1C2C3"
        ) == "hexacyclo[15.3.2.2^3,7.1^2,12.0^13,21.0^11,25]pentacosane"


class TestSpiroPriorityNumbering:
    """The tricyclo+ spiro cage numbers the spiro junction at the LOWEST locant
    among same-descriptor candidates, not the heteroatom-optimal one."""

    def test_spiro_atom_gets_lowest_locant(self):
        mol = Chem.MolFromSmiles(W1)
        res = _name_spiro_component(mol, set(W1_CAGE), W1_SPIRO)
        assert res is not None, "tricyclo+ spiro component should still name"
        name, a2l = res
        #: spiro atom at the lowest achievable locant (3), NOT 13.
        assert a2l[W1_SPIRO] == 3, (name, a2l[W1_SPIRO])
        # descriptor unchanged; only the numbering (hence 'a'-prefix locant) moves
        assert name == "13-oxatricyclo[8.3.0.0^2,6]tridecane", name


class TestZeroWrong:
    """0-wrong ABSOLUTE: the full molecule either abstains or names the RIGHT
    molecule (renumbering never changes constitution, so RT is invariant)."""

    @pytest.mark.opsin_gate
    def test_witness_never_wrong(self):
        name = name_compound(W1)
        assert (
            name is None
            or name.startswith("unknown")
            or opsin_roundtrip_check(W1, name)["passed"]
        ), f"0-wrong violation: {W1} -> {name!r}"
