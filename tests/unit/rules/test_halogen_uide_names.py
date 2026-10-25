"""Halogen -uide anions: every ligand is named by the trusted substituent-prefix namer, the
prefixes are assembled under, and a name ships only when something reads it back.

A halogen -uide name (``diphenyliodanuide``) is a name OPSIN 2.9.0 has no grammar for: it rejects a
substituted ``iodanuide`` / ``bromanuide`` / ``chloranuide`` token ("unphysical valency state"). It used
to ship with NO read-back (the ``halogen_uide`` grammar-gap carve-out), and 111 of the 349 names the
default tier gave for the small PubChem halogen anions were the name of another molecule: a branched
alkyl named as a straight chain (``CC(C)[I-]C`` -> ``methylpropyliodanuide``), unsaturation dropped
(``C[I-]C=C`` -> ``ethylmethyliodanuide``) and a halogen ligand cited without the enclosing marks that
tell it from a substituted alkyl (``C[I-]Cl`` -> ``chloromethyliodanuide`` reads as ClCH2-).

The rules:

* ANIONS FORMED BY ADDITION OF HYDRIDE IONS (the Blue Book): "(1) the suffix 'uide'
  describes an anion formally derived by adding a hydride ion, H-, to a parent hydride name" (:41093),
  "(2) by adding the suffix 'ide' to a parent hydride in which the bonding number at the anionic site is
  higher than the standard bonding number and is expressed by the lambda-convention" (:41094), "Method (1)
  leads to preferred IUPAC names." (:41098); 'diphenyliodanuide (PIN) diphenyl-lambda3-iodanide' (:41110):
  the Blue Book gives the method (2) spelling of the same anion, which OPSIN reads. The default tier ships
  a halogen -uide name when its method (2) spelling reads to the input's FULL InChIKey.
* (the Blue Book), (:7272): "For mononuclear parent hydrides with two or more
  substituents the first cited substituent never has enclosing marks unless it includes a locant. The second
  and further substituents are each enclosed with parentheses even for simple substituents. When the simple
  substituent groups are accompanied by multiplicative prefixes such as 'di' and 'tri', the multiplicative
  prefixes are not included in the parentheses."; 'ethyl(methyl)(propyl)phosphane (PIN)' (:7282),
  'ethyldi(methyl)phosphane (PIN)' (:7290), 'chloro(methyl)silane (PIN)' (:7280), and under
  (:7236) '(chloromethyl)silane [PIN; the name chloro(methyl)silane would describe the structure
  Cl-SiH2-CH3]'. 'difluoro(phenyl)sulfanuide (PIN)' (:41108) is the same assembly on a -uide parent.
* The ligand names: 'propan-2-yl (preferred prefix)' Substituent prefixes derived from
  mononuclear parent hydrides... GENERAL METHODOLOGY FOR NAMING SUBSTITUENT GROUPS,:15946),
  'tert-butyl' Retained prefixes that are preferred prefixes,:16286 'tert-butyldi(methyl)phosphane
  (PIN)'), 'ethenyl' and 'prop-2-en-1-yl' RETAINED NAMES FOR SUBSTITUENT GROUPS DERIVED FROM
  UNSATURATED ACYCLIC PARENT HYDRIDES,:17362: "The systematic names ethenyl, ethenylidene, prop-2-en-1-yl
  ... are the preferred IUPAC names").
"""
import contextlib
import re

import pytest
from rdkit import Chem

from tests.support.default_tier import default_tier_row
from tests.support.opsin_marks import release_uide_only
from tests.support.rt_assert import _canon, _independent_parse, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (smiles, the name the Blue Book rules give); each of these was the name of another molecule
WRONG_BEFORE = [
    ("C[I-]C#C", "ethynyl(methyl)iodanuide", "ethylmethyliodanuide"),
    ("C[I-]C=C", "ethenyl(methyl)iodanuide", "ethylmethyliodanuide"),
    ("C=C[I-]C=C", "diethenyliodanuide", "diethyliodanuide"),
    ("CC(C)[I-]C", "methyl(propan-2-yl)iodanuide", "methylpropyliodanuide"),
    ("CC(C)(C)[I-]C1=CC=CC=C1", "tert-butyl(phenyl)iodanuide", "butylphenyliodanuide"),
    ("C[I-]Cl", "chloro(methyl)iodanuide", "chloromethyliodanuide"),
    ("C[I-]Br", "bromo(methyl)iodanuide", "bromomethyliodanuide"),
    ("CC(C)[I-]I", "iodo(propan-2-yl)iodanuide", "iodopropyliodanuide"),
    ("C=CC[I-]CC=C", "di(prop-2-en-1-yl)iodanuide", "dipropyliodanuide"),
    ("C=C[I-]C1CCN(CC1)I", "ethenyl(1-iodopiperidin-4-yl)iodanuide", "ethyl(1-iodopiperidin-4-yl)iodanuide"),
]
# the names the default tier gave right: they stay, and the spelling follows
RIGHT_BEFORE = [
    ("CC[I-]C", "ethyl(methyl)iodanuide"),                       # was ethylmethyliodanuide: the second group takes marks
    ("c1ccccc1[I-]c1ccccc1", "diphenyliodanuide"),               # the Blue Book
    ("CC(C)(C)C1=CC=C(C=C1)[I-]C2=CC=C(C=C2)C(C)(C)C", "bis(4-tert-butylphenyl)iodanuide"),
    ("C[IH-]", "methyliodanuide"),
    ("C(c1ccccc1)[I-]C", "benzyl(methyl)iodanuide"),
    ("C1CCCC1[I-]C", "cyclopentyl(methyl)iodanuide"),
    ("C[I-]c1ccc(cc1)[N+](=O)[O-]", "methyl(4-nitrophenyl)iodanuide"),   # a neutral group with charged atoms
]
# rows the default tier declined: a ligand that starts the name with a locant or an enclosing mark (the
# carve-out's pattern began with a bare letter) or that carries a heteroatom (the carbon-count namer refused
# it); the first is the decorated mutant of 'diphenyliodanuide', which must never give the bare name
NEWLY_NAMED = [
    ("[I-](c1ccccc1)c1ccc(C)cc1", "(4-methylphenyl)(phenyl)iodanuide"),
    ("[I-](CCO)c1ccccc1", "(2-hydroxyethyl)(phenyl)iodanuide"),
    ("C[I-]C(F)(F)F", "methyl(trifluoromethyl)iodanuide"),
    ("C[I-]C(C)=O", "acetyl(methyl)iodanuide"),
    ("C[I-]OC", "methoxy(methyl)iodanuide"),
    # the best-effort prefix namer gives '(methylamino)amino' for -N=N-CH3 (the double bond dropped), a tier
    # inversion while the default tier ships 'methyldiazenyl': a wider tier names a ligand as the default tier does
    ("C[I-]N=NC", "methyl(methyldiazenyl)iodanuide"),
]
ALL_ROWS = [(s, n) for s, n, _ in WRONG_BEFORE] + RIGHT_BEFORE + NEWLY_NAMED

# a name that is read back by OPSIN directly: the other -uide elements, whose names did not change when
# the ligands moved to the trusted namer, and the ones whose spelling follows
OTHER_STEMS_SAME = [
    ("C[B-](C)(C)C", "tetramethylboranuide"),
    ("F[B-](F)(F)F", "tetrafluoroboranuide"),
    ("[B-](c1cccc(SC)c1)(F)(F)F", "trifluoro[3-(methylsulfanyl)phenyl]boranuide"),
    # a ring ligand with a prefix of its own nested in it keeps the enclosing marks of the ring-substituent
    # chokepoint ('4-trifluoroacetylphenyl' from the five-tier chokepoint would lose the marks and
    # the default tier would decline the name: the name was a pin_verified row at main)
    ("[B-](C1=CC=C(C=C1)C(=O)C(F)(F)F)(C2=CC=C(C=C2)C(=O)C(F)(F)F)(C3=CC=C(C=C3)C(=O)C(F)(F)F)"
     "C4=CC=C(C=C4)C(=O)C(F)(F)F", "tetrakis[4-(trifluoroacetyl)phenyl]boranuide"),
    ("C[SiH4-]", "methylsilanuide"),
    ("C[P-](C)(C)C", "tetramethylphosphanuide"),
]
OTHER_STEMS_RESPELLED = [
    ("CC[B-](C)(C)C", "ethyltri(methyl)boranuide"),              #:7290 'ethyldi(methyl)phosphane (PIN)'
    ("c1ccccc1[B-](F)(F)F", "trifluoro(phenyl)boranuide"),       #:41108 'difluoro(phenyl)sulfanuide (PIN)'
    ("[B-](C1=CC=CC=C1OC2CCCCCC2)(F)(F)F", "[2-(cycloheptyloxy)phenyl]tri(fluoro)boranuide"),
    ("CC[B-](F)(F)F", "ethyltri(fluoro)boranuide"),                #:25866 'chlorodi(methyl)borane (PIN)'
]
# a branched or unsaturated ligand on a -uide centre OPSIN reads: the default tier abstained (the name of
# another molecule was withdrawn by the read-back), now the ligand has its name
OTHER_STEMS_NEWLY_NAMED = [
    ("CC(C)[B-](C)(C)C", "trimethyl(propan-2-yl)boranuide"),
    ("C=C[B-](C)(C)C", "ethenyltri(methyl)boranuide"),
    ("CC(C)(C)[Si-](C)(C)(C)C", "tert-butyltetra(methyl)silanuide"),
]
ONIUM_ROWS = [
    ("C[I+]C=C", "ethenyl(methyl)iodanium"),
    ("CC(C)[I+]C", "methyl(propan-2-yl)iodanium"),
    ("C[I+]C", "dimethyliodanium"),
    ("c1ccccc1[I+]c1ccccc1", "diphenyliodanium"),
    ("C[Cl+]C(C)=O", "acetyl(methyl)chloranium"),
    # best-effort kept the name of these and lowered its label (pin_verified at the default tier): the general
    # engine named the cation capped with a stand-in -OH, '(hydroxyethynyl)(phenyl)iodanium', and its
    # string won; the stand-in is no ligand of the cation
    ("CC1=CC=C(C=C1)C#C[I+]C2=CC=CC=C2", "[2-(4-methylphenyl)ethynyl](phenyl)iodanium"),
    ("C1=CC=C(C=C1)[I+]C#CC2=CSC=C2", "phenyl[2-(thiophen-3-yl)ethynyl]iodanium"),
    ("OCC[I+]C", "(2-hydroxyethyl)(methyl)iodanium"),
    # the ring-substituent chokepoint gives '4-(octyloxy)phenyl' for this ligand (the carbons of the second ring
    # counted as a chain); the five-tier chokepoint gives the right prefix, and best-effort abstained
    ("CCC1(CCCCC1)OC2=CC=C(C=C2)[I+]C3=CC=CC=C3", "{4-[(1-ethylcyclohexyl)oxy]phenyl}(phenyl)iodanium"),
]


def _neutral_twin_reads_back(name, smiles):
    """An independent read-back of a halogen -uide name: the centre made neutral with one more hydrogen is
    the structure of the neutral ``lambda<n>``-iodane (n = the valence of the neutral centre), which OPSIN
    reads, so the name with its parent token ``iodanuide`` spelled as that iodane reads to the full
    InChIKey of the input with the centre made neutral. A different spelling from the one the engine
    checks (method (2), ``-lambda<n>-iodanide``), compared on another structure."""
    m = re.fullmatch(r"(?P<prefix>\S+?)(?P<stem>iod|brom|chlor)anuide", name)
    if m is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    centres = [a for a in mol.GetAtoms() if a.GetFormalCharge() == -1
               and a.GetSymbol() in ("I", "Br", "Cl") and a.GetTotalDegree() >= 2]
    if len(centres) != 1:
        return False
    centre = centres[0]
    valence = int(sum(b.GetBondTypeAsDouble() for b in centre.GetBonds()) + centre.GetTotalNumHs())
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(centre.GetIdx())
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(centre.GetTotalNumHs() + 1)
    atom.SetNoImplicit(True)
    Chem.SanitizeMol(rw)
    twin = f"{m.group('prefix')}-λ{valence + 1}-{m.group('stem')}ane"
    parsed = _independent_parse(twin, Chem.MolToSmiles(rw))
    if parsed is None or Chem.MolFromSmiles(parsed) is None:
        return False
    return Chem.MolToInchiKey(rw) == Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))


@release_uide_only
@pytest.mark.parametrize("smiles,name", ALL_ROWS)
def test_default_tier_gives_the_name_and_reads_it_back_through_its_twin(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin_twin" and row["opsin"] == "verified_twin", row
    assert row["gate_outcome"] == "twin_verified:uide_lambda", row
    assert row["spelling_failures"] == [], row


@release_uide_only
@pytest.mark.parametrize("smiles,name", ALL_ROWS)
def test_best_effort_names_every_row_the_default_tier_names(smiles, name):
    """A tier inversion: the default tier named these rows and best-effort abstained (suppressed)."""
    row = name_best_effort(smiles)
    assert row["name"] == name, row
    assert row["verified"] == "opsin_twin" and row["gate_outcome"] == "twin_verified:uide_lambda", row


@pytest.mark.parametrize("smiles,name", ALL_ROWS)
def test_the_neutral_iodane_twin_reads_the_name_to_the_input(smiles, name):
    """The check the engine does not make: another spelling of the same structure."""
    assert _neutral_twin_reads_back(name, smiles)


@pytest.mark.parametrize("smiles,name,wrong", WRONG_BEFORE)
def test_the_wrong_name_is_the_name_of_another_molecule(smiles, name, wrong):
    """The wrong names are read by the same twin: they do not read to the input."""
    assert not _neutral_twin_reads_back(wrong, smiles)
    assert _neutral_twin_reads_back(name, smiles)


@pytest.mark.parametrize("smiles,name,wrong", WRONG_BEFORE)
def test_no_tier_ships_the_wrong_name(smiles, name, wrong):
    assert default_tier_row(smiles)["name"] != wrong
    assert name_best_effort(smiles)["name"] != wrong


def test_a_halogen_uide_name_the_twin_does_not_read_is_declined(monkeypatch):
    """No -uide name ships unless something reads it back: a producer that builds the old wrong name for
    ``C[I-]C#C`` is declined at the default tier (and the wrong name is not shipped at best-effort)."""
    import orthonym.rules.ions as ions
    monkeypatch.setattr(ions, "_emit_group13_uide", lambda mol, idx: "ethylmethyliodanuide")
    row = default_tier_row("C[I-]C#C")
    assert row["tier"] == "abstain" and row["name"] != "ethylmethyliodanuide", row
    assert name_best_effort("C[I-]C#C")["name"] != "ethylmethyliodanuide"


def test_a_ligand_with_a_net_charge_gives_no_uide_name():
    """A cationic ligand makes the compound a zwitterion, not an -uide anion with a prefix:
    "4 Anions", "5 Zwitterions", the Blue Book)."""
    row = default_tier_row("C[I-]C[N+](C)(C)C")
    assert row["tier"] == "abstain" and not row["name"].endswith("iodanuide"), row
    assert not (name_best_effort("C[I-]C[N+](C)(C)C")["name"] or "").endswith("iodanuide")


def test_a_ligand_without_a_name_declines_the_emitter(monkeypatch):
    """A ligand the trusted namer cannot name gives no name (never a carbon count)."""
    import orthonym.assembly.substituent_enumerator as enumerator
    import orthonym.rules.ions as ions
    mol = Chem.MolFromSmiles("C[I-]C#C")
    centre = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "I")
    assert ions._emit_group13_uide(mol, centre) == "ethynyl(methyl)iodanuide"
    monkeypatch.setattr(enumerator, "name_substituent", lambda *a, **k: "substituent")
    assert ions._emit_group13_uide(mol, centre) == ""


@pytest.mark.parametrize("smiles,name", OTHER_STEMS_SAME + OTHER_STEMS_RESPELLED + OTHER_STEMS_NEWLY_NAMED)
def test_other_uide_stems_are_named_and_read_by_opsin(smiles, name):
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert row["name"] == name and row["tier"] == "pin_verified", row
        assert row["verified"] == "opsin" and row["spelling_failures"] == [], row
    assert _canon(_independent_parse(name, smiles)) == _canon(smiles)


@pytest.mark.parametrize("smiles,name", ONIUM_ROWS)
def test_halogen_onium_names_read_back_to_the_input(smiles, name):
    """``emit_halogen_onium`` named its ligands by carbon count as well (``C[I+]C=C`` -> ethyl)."""
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert row["name"] == name and row["tier"] == "pin_verified", row
    assert _canon(_independent_parse(name, smiles)) == _canon(smiles)


@contextlib.contextmanager
def _breadth_contexts():
    """The contexts a best-effort call publishes (``Orthonym.name``), for a producer called directly."""
    from orthonym.metrics.provenance import (allow_aromatic_general_ctx, best_effort_ctx, full_coverage_ctx,
                                             general_fallback_ctx)
    breadth = (general_fallback_ctx, best_effort_ctx, allow_aromatic_general_ctx, full_coverage_ctx)
    tokens = [ctx.set(True) for ctx in breadth]
    try:
        yield
    finally:
        for ctx, token in zip(breadth, tokens):
            ctx.reset(token)


def test_a_stand_in_cap_is_no_ligand_of_an_ion_centre():
    """The capped-fragment probe (``substituent_enumerator._route_fragment_to_general_engine``) caps a free
    valence with -OH and names the capped molecule with the contexts of a best-effort call. Under the probe a chain ligand with an atom that is no carbon
    declines (the ion outranks the cap,, the Blue Book, so the capped name cannot become
    a prefix: the string the general engine built from it won at best-effort, at a lower label); a ring
    ligand and a real molecule keep their names."""
    from orthonym.rules.centre_ligands import capped_probe_ctx
    from orthonym.rules.ions import emit_halogen_onium
    capped, ring, real = (Chem.MolFromSmiles(s) for s in ("OC#C[I+]c1ccccc1", "Oc1ccc(cc1)[I+]C", "OCC[I+]C"))

    def named(mol):
        return emit_halogen_onium(mol, next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "I"))

    # (the Blue Book): the hydroxy group can only be on C2 of an ethynyl ligand, so the
    # group cites no locant ('chlorodiazenyl (preferred prefix)',:2987); the default tier names the
    # ligand ('(2-hydroxyethyn-1-yl)' was not a spelling it could ship before, and abstained here)
    assert named(capped) == "(hydroxyethynyl)(phenyl)iodanium"
    with _breadth_contexts():
        assert named(capped) == "(hydroxyethynyl)(phenyl)iodanium"
        token = capped_probe_ctx.set(True)
        try:
            assert named(capped) == ""
            assert named(ring) == "(4-hydroxyphenyl)(methyl)iodanium"
            assert named(real) == ""        # what the cap of a probe looks like to the emitter
        finally:
            capped_probe_ctx.reset(token)
        assert named(real) == "(2-hydroxyethyl)(methyl)iodanium"


def test_a_wider_tier_falls_back_to_the_default_tiers_ligand_name():
    """No ligand namer is right for every ligand, so a wider tier takes the first of the two tiers' ligand names
    that reads back: the best-effort namer's '(methylamino)amino' for -N=N-CH3 does not, the default tier's
    'methyldiazenyl' does; the default tier's '4-decylphenyl' for 4-(adamantan-1-yl)phenyl does not, the
    best-effort ring namer's prefix does (rows of the onium panel were lost when the default tier's names were
    taken first)."""
    from orthonym.rules.centre_ligands import centre_ligand_prefixes, centre_name
    mol = Chem.MolFromSmiles("C[I-]N=NC")
    centre = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "I")
    assert centre_ligand_prefixes(mol, centre) == ["methyl", "methyldiazenyl"]
    with _breadth_contexts():
        assert centre_ligand_prefixes(mol, centre) == ["methyl", "(methylamino)amino"]
        assert centre_name(mol, centre, "iodanuide", "iodanuide") == "methyl(methyldiazenyl)iodanuide"
    assert centre_name(mol, centre, "iodanuide", "iodanuide") == "methyl(methyldiazenyl)iodanuide"


def test_a_wider_tier_keeps_a_ring_prefix_the_default_tier_misnames():
    row = name_best_effort("C1C2CC3CC1CC(C2)(C3)C4=CC=C(C=C4)[I+]C5=CC=CC=C5")
    assert row["name"] == "[4-(adamantan-1-yl)phenyl](phenyl)iodanium" and row["verified"] == "opsin", row
    assert _canon(_independent_parse(row["name"], "C1C2CC3CC1CC(C2)(C3)C4=CC=C(C=C4)[I+]C5=CC=CC=C5")) == _canon(
        "C1C2CC3CC1CC(C2)(C3)C4=CC=C(C=C4)[I+]C5=CC=CC=C5")


def test_only_the_uide_twin_outcome_passes_a_general_tier():
    """The two general-tier checks of ``name_tiered`` (the exit check and the full-key check of the label)
    let a name pass on a twin outcome only for ``twin_verified:uide_lambda``: the other twin families spell
    the form OPSIN reads at a wider tier, so a twin outcome of theirs there is no read-back of the shipped
    string. The one other outcome they pass is ``carveout:phane`` (CO lane P, lead decision 2026-10-17: best-effort
    names the phane rows the default tier names): a general tier records it only for a phane name that
    ``validation.phane_read_back`` builds to the input (``Orthonym._last_resort_list_name``, after the engine has no
    name), and no other carve-out passes."""
    import inspect

    import orthonym.namer as namer
    from orthonym.metrics import provenance as pv
    from orthonym.validation.uide_twin import uide_twin_outcome
    assert uide_twin_outcome("twin_verified:uide_lambda") and uide_twin_outcome("carveout:phane")
    for other in ("twin_verified:isotope_count_one", "twin_verified:bridge_sulfano", "twin_verified:uide_lambda+x",
                  "self_consistency_verified", "carveout:np_stereoparent", "carveout:polyol_polyester",
                  "carveout:organometallic_additive", "carveout:dianhydride", "carveout:thioperoxol",
                  "carveout:phane+x", "", None):
        assert not uide_twin_outcome(other), other
    # the exit check of a best-effort call: a name OPSIN cannot read and an isotope twin outcome is withdrawn
    name, smiles = "ethan(2H)ol", "[2H]OCC"
    best_effort = namer.Orthonym(**__import__("orthonym.cli", fromlist=["_emit_tier_flags"])._emit_tier_flags("best-effort"))
    pv.clear_provenance()
    pv.record_gate_outcome("twin_verified:isotope_count_one", name)
    assert best_effort._exit_round_trip_check_rt(name, smiles) != name
    pv.clear_provenance()
    pv.record_gate_outcome("twin_verified:uide_lambda", name)
    assert best_effort._exit_round_trip_check_rt(name, smiles) == name
    pv.clear_provenance()
    # the label check keeps the full-key read-back for every other outcome
    source = inspect.getsource(namer.Orthonym._tier_row_and_pin_form)
    assert 'opsin != "verified" and not uide_twin_outcome(gate_outcome)' in source
    assert "opsin not in _pv.VERIFIED_OPSIN_LABELS and name" not in source
