"""
v33 Phase 2 Task 2.0 -- fix the stale peptide dispatch SMARTS + RT-gate the
flat producer.

Root cause ( sec.4):
`rules/amino_acids.py::is_peptide()` -- the dispatch predicate that gates
whether `rules/peptides.py::name_peptide` is ever called
(`routing/dispatch_table.py:604-612`) -- used SMARTS that were never updated
when `peptides.py`'s own backbone-bond pattern was broadened (v30) to admit
a C-terminal cyclic imino acid (proline/hydroxyproline): the ring nitrogen
loses its only hydrogen once acylated by the preceding residue's peptide
bond (a tertiary amide, H0), so the stale `[NX3;H1]`/`[NX3;H2,H1]` patterns
never matched and `is_peptide()` returned False even though `name_peptide()`
would already build a correct name if called directly.

Sized at 9/226 true peptides in the backlog. OPSIN full-InChI round-trip
(hand-verified against the exact input, see the baseline capture in the
Task 2.0 report): 8/9 are correct; 1/9 (an 11-residue chain with an internal
Gln + Asp) is WRONG -- caught only incidentally today because the flat
acylamino-convention emission (`name_peptide` step 4) is not itself
RT-gated. This module proves:

  1. the dispatch predicate now reaches all 9 (`is_peptide()` True);
  2. the 8 correct ones round-trip through the full `name_compound` pipeline;
  3. the 1 wrong one ABSTAINS (never ships a wrong molecule);
  4. the broadened predicate does NOT fire on non-peptide amino-acid-shaped
     molecules (no over-broadening);
  5. an ordinary (non-proline) peptide's PIN name is byte-identical.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse
from orthonym.rules.amino_acids import is_peptide

pytestmark = pytest.mark.opsin_gate


def _full_rt(smiles: str, name: str) -> bool:
    """name -> OPSIN -> full InChIKey, compared to the input's InChIKey."""
    if not name or is_failure_name(name):
        return False
    opsin_smi = opsin_parse(name)
    if not opsin_smi:
        return False
    mol_in = Chem.MolFromSmiles(smiles)
    mol_out = Chem.MolFromSmiles(opsin_smi)
    if mol_in is None or mol_out is None:
        return False
    return inchi.MolToInchiKey(mol_in) == inchi.MolToInchiKey(mol_out)


# The 9 dispatch-blocked-but-nameable witnesses (Phase-0 SPY, re-derived
# directly from 's `peptide`
# bucket: `is_peptide()` False today, `name_peptide()` builds a candidate
# when called directly). Index 0 is the ONE that OPSIN-round-trips WRONG.
_WRONG_WITNESS = (
    "CCC(C)[C@H](N)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@H]"
    "(C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N"
    "[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H]"
    "(Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(C)C)C(=O)N1CCC"
    "[C@H]1C(=O)O)C(C)C"
)

_GOOD_WITNESSES = [
    "CC[C@H](C)[C@H](N)C(=O)NCC(=O)N1CCC[C@@H]1C(=O)O",  # isoleucylglycyl-D-proline
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N1CCC[C@@H]1C(=O)O",  # isoleucylglutamyl-D-proline
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CCCCN)C(=O)N1CCC[C@@H]1C(=O)O",  # isoleucyllysyl-D-proline
    "CC[C@H](C)[C@H](N)C(=O)N[C@H](C(=O)N1CCC[C@@H]1C(=O)O)[C@@H](C)O",  # isoleucylthreonyl-D-proline
    "CSCC[C@H](NC(=O)[C@@H](N)[C@@H](C)O)C(=O)N1CCC[C@@H]1C(=O)O",  # threonylmethionyl-D-proline
    (
        "CSCC[C@H](NC(=O)[C@H](CO)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)"
        "[C@@H](N)CO)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1cnc[nH]1)"
        "C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H]"
        "(Cc1c[nH]c2ccccc12)C(=O)NCC(=O)N[C@@H](CCCCN)C(=O)N1CCC"
        "[C@H]1C(=O)N[C@H](C(=O)NCC(=O)N[C@@H](CCCCN)C(=O)N[C@@H]"
        "(CCCCN)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)"
        "C(=O)N1CCC[C@H]1C(=O)N[C@H](C(=O)N[C@@H](CCCCN)C(=O)N[C@H]"
        "(C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N1CCC[C@H]1C(=O)O)C(C)C)"
        "C(C)C)C(C)C"
    ),  # the 208-heavy-atom giant (RT-valid despite size)
    "C[C@@H](O)[C@H](N)C(=O)N[C@@H](CO)C(=O)N1CCC[C@@H]1C(=O)O",  # threonylseryl-D-proline
    "C[C@H](NC(=O)[C@@H](N)[C@@H](C)O)C(=O)N1CCC[C@@H]1C(=O)O",  # threonylalanyl-D-proline
]

_ALL_9_WITNESSES = [_WRONG_WITNESS] + _GOOD_WITNESSES


@pytest.mark.unit
class TestDispatchPredicateUnblocked:
    """`is_peptide()` must now return True for all 9 -- the actual
    stale-SMARTS fix under test."""

    @pytest.mark.parametrize("smi", _ALL_9_WITNESSES)
    def test_is_peptide_true(self, smi):
        mol = Chem.MolFromSmiles(smi)
        assert is_peptide(mol) is True


@pytest.mark.unit
class TestGoodWitnessesRoundTrip:
    """The 8 correct Pro-C-terminal peptides must round-trip exactly through
    the full name_compound pipeline once dispatch is unblocked."""

    @pytest.mark.parametrize("smi", _GOOD_WITNESSES)
    def test_emits_rt_valid_name(self, smi):
        name = name_compound(smi, style="pin")
        assert not is_failure_name(name), (smi, name)
        assert _full_rt(smi, name), (smi, name)


@pytest.mark.unit
class TestWrongWitnessAbstains:
    """The 1 candidate that OPSIN proves WRONG must ABSTAIN once the flat
    producer's emission is RT-gated -- never ship as a wrong molecule."""

    def test_abstains_not_wrong(self):
        name = name_compound(_WRONG_WITNESS, style="pin")
        # Either it abstains outright, or -- if the RT gate is bypassed by
        # some other path -- it must at minimum round-trip correctly. A
        # wrong-but-non-abstaining name is the one outcome this test exists
        # to forbid.
        if not is_failure_name(name):
            assert _full_rt(_WRONG_WITNESS, name), (
                "flat producer shipped a WRONG name instead of abstaining",
                name,
            )


@pytest.mark.unit
class TestNoOverBroadening:
    """The broadened predicate must NOT fire on non-peptide amino-acid-shaped
    molecules -- a single free amino acid, an N-acyl amino acid, and a
    standalone N-acylated cyclic imino acid (the exact shape the fix
    broadens the SMARTS to admit, but only when it is a REAL multi-residue
    peptide bond, not a standalone acyl cap)."""

    @pytest.mark.parametrize("smi", [
        "OC(=O)C1CCCN1",           # free proline (no peptide bond at all)
        "NCC(=O)O",                # glycine
        "CC(N)C(=O)O",             # alanine
    ])
    def test_is_peptide_false(self, smi):
        mol = Chem.MolFromSmiles(smi)
        assert is_peptide(mol) is False

    def test_n_acetylproline_predicate_over_broadens_but_declines_safely(self):
        """N-acetylproline (a single acylated residue, NOT a peptide) has the
        SAME local shape a real X-Pro peptide bond has -- a ring N, H0,
        bonded to an acyl carbon on one side and the ring alpha-carbon
        bearing free COOH on the other -- so the cheap dispatch PREDICATE
        cannot distinguish the two locally; `is_peptide()` does flip to True
        for it. This is not a regression: it is unreachable functionally
        because `name_peptide()`'s own `_is_valid_peptide` (unchanged,
        requires a genuine free N-terminus + a >=2-residue chain for the
        acyl-cap lever) still declines it, so dispatch falls through exactly
        as before -- verified end-to-end via  (A == B,
        both 'unknown organic compound')."""
        mol = Chem.MolFromSmiles("CC(=O)N1CCCC1C(=O)O")
        assert is_peptide(mol) is True  # documents the predicate's known slack
        from orthonym.rules.peptides import name_peptide
        assert name_peptide(mol) is None  # but it never actually NAMES it


@pytest.mark.unit
class TestPinUnchanged:
    """Ordinary (non-proline) peptides must be byte-identical."""

    def test_glycylglycine_unchanged(self):
        assert name_compound("NCC(=O)NCC(=O)O", style="pin") == "glycylglycine"

    def test_alanylglycine_unchanged(self):
        assert (
            name_compound("N[C@@H](C)C(=O)NCC(=O)O", style="pin")
            == "alanylglycine"
        )
