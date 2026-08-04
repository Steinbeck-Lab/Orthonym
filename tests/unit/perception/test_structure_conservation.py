"""Wave-8 P10 -- Tier-B structure-conservation checks (Java-free).

Covers ``perception/structure_conservation.py``: the input invariants helper
(P10.1) and the two source-level vetoes wired into
``namer.py::Orthonym._name_impl`` (charge-conservation, R12-spillover
sp3-ring substituent drop on the partially-saturated fused-carbocycle
emitter). See the module docstring for the full design record.
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.perception.structure_conservation import (
    input_invariants,
    net_formal_charge,
    looks_like_ionic_name,
    charge_dropped,
    partial_sat_sp3_substituent_drop,
)


# ---------------------------------------------------------------------------
# input_invariants (P10.1)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_input_invariants_exocyclic_terpene():
    # C=C1CCC2CCC1C2: 9 heavy atoms, 2 rings + 1 exocyclic C=C = DoU 3.
    mol = Chem.MolFromSmiles("C=C1CCC2CCC1C2")
    heavy, dou, _bond_order_sum = input_invariants(mol)
    assert heavy == 9
    assert dou == 3


@pytest.mark.unit
def test_input_invariants_fragment_drop_glycoside():
    mol = Chem.MolFromSmiles(
        "O=c1cc(-c2ccc(O[C@@H]3OC(CO)[C@@H](O)[C@H](O)C3O)cc2)"
        "oc2cc(O)cc(O)c12"
    )
    heavy, _dou, _bond_order_sum = input_invariants(mol)
    assert heavy == mol.GetNumHeavyAtoms()  # authoritative, not estimated


@pytest.mark.unit
def test_input_invariants_benzene_ring_only():
    mol = Chem.MolFromSmiles("c1ccccc1")
    heavy, dou, _ = input_invariants(mol)
    assert heavy == 6
    assert dou == 4  # 1 ring + 3 aromatic "double" bonds (Kekule count)


@pytest.mark.unit
def test_net_formal_charge():
    assert net_formal_charge(Chem.MolFromSmiles("CCO")) == 0
    assert net_formal_charge(Chem.MolFromSmiles("[O-]C(=O)C")) == -1
    assert net_formal_charge(Chem.MolFromSmiles("[NH4+]")) == 1
    assert net_formal_charge(Chem.MolFromSmiles("[O-]S(=O)(=O)[O-]")) == -2


# ---------------------------------------------------------------------------
# Charge-conservation veto
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("name", [
    "sulfate", "phosphate", "hexan-3-ide", "ethanethioate",
    "acetylazanide", "methylsilanuide", "tetramethylphosphanuide",
    "N,N,N-trimethylmethanaminium", "azanylium", "silylium",
    "benzenediazonium", "ethylsulfanylium",
    "ethane-1,2-bis(aminium)", "ethane-1,2-bis(olate)",
    "ethane-1,2-bis(thiolate)", "ethane-1,2-bis(aminide)",
    "2-(carboxylatomethyl)benzoate",
])
def test_looks_like_ionic_name_recognises_known_charged_pins(name):
    assert looks_like_ionic_name(name) is True


@pytest.mark.unit
@pytest.mark.parametrize("name", [
    "ethanol", "2-iodobenzenylethan-1-ol", "trimethylsilane", "benzene", "",
])
def test_looks_like_ionic_name_rejects_neutral_shapes(name):
    assert looks_like_ionic_name(name) is False


@pytest.mark.unit
@pytest.mark.parametrize("smiles,name", [
    # documented leak: net-charged input, neutral-shaped WRONG-CONNECTIVITY
    # shipped name (OPSIN: '2-iodobenzenylethan-1-ol' = 1-(2-iodophenyl)ethanol,
    # a different molecule). charge_dropped fires (hypervalent halogen centre).
    ("[I-](CCO)c1ccccc1", "2-iodobenzenylethan-1-ol"),
    # NOTE (W8-P10 controller 2026-07-18): C[Si-](C)(C)[H] -> 'trimethylsilane'
    # is intentionally NOT a charge_dropped leak here. It is the established
    # over-coordinated-group-14-metalloid class, DELIBERATELY named as its
    # neutral parent (test_charged_suffixes_ft6.py::
    # test_overcoordinated_silicon_has_no_uide). charge_dropped is narrowed to
    # hypervalent HALOGENS (a wrong-CONNECTIVITY drop), so it does not fire on
    # a metalloid charge-NORMALIZATION; that stays a gated-safe residual
    # (production is already 'unknown' via SELF-01).
])
def test_charge_dropped_true_for_documented_leaks(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    assert charge_dropped(mol, name) is True


@pytest.mark.unit
def test_charge_dropped_known_limitation_boron_olate_coincidence():
    """KNOWN, DOCUMENTED GAP (precision-over-recall; not closed by design).

    ``[B-](CCO)(C)(C)C`` raw-names as '2-(trimethylboryl)ethan-1-olate' --
    the boron anion is dropped (named as neutral 'boryl'), but the string
    coincidentally ends in '-olate', a genuine IUPAC anion suffix (it just
    happens to belong to the wrong atom -- the terminal CH2-OH, not the
    boron). The boron centre IS in scope
    (``_has_main_group_charge_centre`` returns True), but the NAME-shape
    guard cannot distinguish this from a correctly-named alkoxide anion
    without atom-level correlation between the charge site and the name,
    which could not be built here without risking new false positives
    elsewhere (see the phase report). Left un-closed intentionally: this
    asserts the CURRENT (gap) behaviour so a future fix is a deliberate,
    reviewed change, not a silent regression.
    """
    mol = Chem.MolFromSmiles("[B-](CCO)(C)(C)C")
    assert charge_dropped(mol, "2-(trimethylboryl)ethan-1-olate") is False


@pytest.mark.unit
@pytest.mark.parametrize("smiles,name", [
    # OVER-VETO GUARD: routine ionic centres must NEVER trigger the veto even
    # when the name lacks an ionic suffix -- these are the choline/ammonium
    # (N+) and carboxylate/phenolate (O-) functional-class shapes a corpus
    # probe flagged as the over-veto risk. Charge centre is N/O -> excluded.
    ("C[N+](C)(C)CCOP(=O)(O)OC[C@H](O)CO",
     "choline (2R)-2-hydroxy-3-phosphonooxypropan-1-ol"),
    ("[NH3+]Cc1ccccc1", "azaniumylmethylbenzene"),      # N+ centre, 'benzene' ending
    ("CCCCCCC([NH3+])C(=O)O", "2-azaniumyloctanoic acid"),  # N+ centre, 'acid' ending
])
def test_charge_dropped_excludes_routine_ionic_centres(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    # sanity: these really are net-charged (so only the centre-scope guard
    # is what spares them, not the net-charge short-circuit)
    assert net_formal_charge(mol) != 0
    assert charge_dropped(mol, name) is False


@pytest.mark.unit
@pytest.mark.parametrize("smiles,name", [
    ("[O-]S(=O)(=O)[O-]", "sulfate"),
    ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),
    ("CCC[CH-]CC", "hexan-3-ide"),
    ("[N+](#N)c1ccccc1", "benzenediazonium"),  # class_id=GENERAL, still ionic-shaped
    ("CC[S+]", "ethylsulfanylium"),            # class_id=GENERAL, still ionic-shaped
    ("[NH3+]CC[NH3+]", "ethane-1,2-bis(aminium)"),
])
def test_charge_dropped_false_for_correct_charged_names(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    assert charge_dropped(mol, name) is False


@pytest.mark.unit
def test_charge_dropped_false_for_neutral_input():
    # Net charge 0 (e.g. a zwitterion or a plain neutral) never triggers,
    # regardless of the name's shape.
    mol = Chem.MolFromSmiles("CC(=O)[O-]")  # placeholder net -1 check below
    assert net_formal_charge(mol) != 0
    neutral_mol = Chem.MolFromSmiles("CCO")
    assert charge_dropped(neutral_mol, "ethanol") is False


@pytest.mark.unit
def test_charge_dropped_false_for_empty_name():
    mol = Chem.MolFromSmiles("[O-]C(=O)C")
    assert charge_dropped(mol, "") is False


# ---------------------------------------------------------------------------
# R12-spillover veto (partial_sat sp3-ring substituent drop)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,bad_name", [
    # Still a leak: no producer can name a 2-aminotetralin (the amine wants a
    # SUFFIX, `1,2,3,4-tetrahydronaphthalen-2-amine`, BlueBookV2.md:26489 shows
    # the 1-isomer as PIN), so the hydro-fused producer fails closed and this
    # veto is what keeps the Java-free path from shipping a substitute.
    ("NC1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalene"),
])
def test_partial_sat_drop_true_for_documented_leaks(smiles, bad_name):
    mol = Chem.MolFromSmiles(smiles)
    assert partial_sat_sp3_substituent_drop(mol, bad_name) is True


@pytest.mark.unit
@pytest.mark.parametrize("smiles,pin", [
    # WAS asserted as leaks that this veto had to SUPPRESS. Both are now named
    # correctly at the source, which is strictly better than being vetoed: the
    # hydro-fused producer became the single speller of every ring substituent
    # prefix (P-16.3.3 multiplicity cannot be split across two formatters) and
    # grew the P-63.1 ring '-ol' suffix. A veto that still fired here would be
    # suppressing the correct answer.
    #
    # Independent check (OPSIN 2.9.0, not the code under test) -- each name
    # parses back to its own input:
    #   '2,7-dimethyl-1,2,3,4-tetrahydronaphthalene' -> Cc1ccc2c(c1)CC(C)CC2
    #   '1,2,3,4-tetrahydronaphthalen-2-ol'          -> OC1CCc2ccccc2C1
    # The old expectations asserted an ABSTENTION, so they cannot have been
    # encoding these molecules' correct names.
    ("Cc1ccc2c(c1)CC(C)CC2", "2,7-dimethyl-1,2,3,4-tetrahydronaphthalene"),
    ("OC1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalen-2-ol"),
])
def test_former_leaks_are_now_named_at_the_source(smiles, pin):
    mol = Chem.MolFromSmiles(smiles)
    assert partial_sat_sp3_substituent_drop(mol, pin) is False
    # and Java-free, so the fix is in the producer rather than in the gate
    raw = Orthonym(style="pin", _disable_opsin_validity_gate=True)
    assert raw.name(smiles) == pin


@pytest.mark.unit
@pytest.mark.parametrize("smiles,good_name", [
    # VALUE CORRECTED (was '1-methyl-...' for this input -- see the note in
    # test_veto_does_not_regress_correct_names). Note this row is now doing
    # real work in the OPPOSITE direction: a veto that fired on the CORRECT
    # `2-methyl-` name would suppress the very fix.
    ("CC1CCc2ccccc2C1", "2-methyl-1,2,3,4-tetrahydronaphthalene"),
    ("c1ccc2c(c1)CCCC2", "1,2,3,4-tetrahydronaphthalene"),
    ("Cc1ccc2c(c1)CCCC2", "6-methyl-1,2,3,4-tetrahydronaphthalene"),
])
def test_partial_sat_drop_false_for_correct_names(smiles, good_name):
    mol = Chem.MolFromSmiles(smiles)
    assert partial_sat_sp3_substituent_drop(mol, good_name) is False


@pytest.mark.unit
def test_partial_sat_drop_false_for_non_matching_class():
    # A molecule this emitter never fires on (e.g. plain hexane) is a no-op.
    mol = Chem.MolFromSmiles("CCCCCC")
    assert partial_sat_sp3_substituent_drop(mol, "hexane") is False


@pytest.mark.unit
def test_partial_sat_drop_false_for_mixed_hydro_shape():
    # A hexahydronaphthalene (mixed alkene/sp3 ring, NOT the clean
    # fully-aromatic + fully-saturated tetralin split) must be excluded --
    # this is the false-positive class the scope guard exists for.
    smiles = "CC1=CC[C@@]23[C@@H]([C@@H]12)[C@H](C(C)C)CC[C@H]3C"
    mol = Chem.MolFromSmiles(smiles)
    name = "(1R,4S,4aR,5S,8aR)-6-methyl-1,2,3,4,4a,5,8,8a-octahydronaphthalene"
    assert partial_sat_sp3_substituent_drop(mol, name) is False


# ---------------------------------------------------------------------------
# End-to-end: the vetoes wired into namer.py fail closed on the raw (no-Java)
# path AND leave the DEFAULT gated namer unchanged on correct names.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "[I-](CCO)c1ccccc1",  # hypervalent-halogen charge-drop (wrong connectivity)
    # NOTE: [B-](CCO)(C)(C)C and C[Si-](C)(C)[H] intentionally excluded --
    # charge_dropped is narrowed to hypervalent halogens; the group-13/14
    # metalloid charge-NORMALIZATIONs (borate-olate coincidence; over-coordinated
    # silane/stannane named neutral per test_charged_suffixes_ft6.py) are
    # gated-safe no-Java residuals, NOT vetoed here (precision-over-recall).
    # R12: the amine wants a SUFFIX no producer builds yet, so the hydro-fused
    # producer fails closed and this veto keeps the Java-free path from
    # shipping the substitute it otherwise reaches (measured:
    # `4-butylcyclohexan-1-amine` -- the aromatic half re-spelled as a butyl
    # chain on a monocycle, a wrong molecule).
    "NC1CCc2ccccc2C1",
    # `Cc1ccc2c(c1)CC(C)CC2` and `OC1CCc2ccccc2C1` MOVED OUT of this list:
    # both are now named correctly at the source, see
    # test_former_leaks_are_now_named_at_the_source.
])
def test_veto_fails_closed_on_documented_leaks_no_java(smiles):
    raw = Orthonym(style="pin", _disable_opsin_validity_gate=True)
    assert raw.name(smiles) == "unknown organic compound"


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C1CC2CCC1C=C2", "bicyclo[2.2.2]oct-2-ene"),
    ("C=C1CCCCC1", "methylidenecyclohexane"),
    # VALUE CORRECTED (was '1-methyl-...', structurally impossible for this
    # input: locants 1 and 4 of `1,2,3,4-tetrahydronaphthalene` are the sp3
    # carbons adjacent to fusion carbons 8a/4a, and this methyl-bearing carbon
    # has no aromatic neighbour. OPSIN 2.9.0: '1-methyl-...' -> CC1CCCc2ccccc21,
    # a DIFFERENT molecule; '2-methyl-...' -> CC1CCc2ccccc2C1 == input.)
    ("CC1CCc2ccccc2C1", "2-methyl-1,2,3,4-tetrahydronaphthalene"),
    ("[O-]S(=O)(=O)[O-]", "sulfate"),
    ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),
])
def test_veto_does_not_regress_correct_names(smiles, expected):
    gated = Orthonym(style="pin")
    assert gated.name(smiles) == expected
