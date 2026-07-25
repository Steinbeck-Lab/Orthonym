"""P-67 organo-oxoacids of the heavier pnictogens (As, Sb).

An arsonic acid is NOT an organometallic -- it is a P-67 oxoacid, the exact
analogue of a phosphonic acid.  The Blue Book gives all four as *preselected
names* (BB L36051-36054):

    AsH(O)(OH)2  arsonic acid    AsH2(O)OH  arsinic acid
    SbH(O)(OH)2  stibonic acid   SbH2(O)OH  stibinic acid

and gives the substituent-prefix PIN forms verbatim:

    CH3-P(O)(OH)2            methylphosphonic acid       (PIN)  BB L36062
    CH3-CH2-SbH(O)OH         ethylstibinic acid          (PIN)  BB L36064
    C6H5-As(CH3)(O)OH        methyl(phenyl)arsinic acid  (PIN)  BB L36066
    (4-acetamido-3-methylphenyl)arsonic acid             (PIN)  BB L33010

Bismuth has NO oxoacid analogue in the Blue Book (there is no "bismuthonic
acid"), so the family is bounded to P / As / Sb -- matching the existing
``_PNICTOGEN_ACID_STEM`` table in ``rules.phosphorus``.

Every expected name in this module was verified OPSIN-exact (name -> SMILES ->
canonical SMILES == canonical SMILES of the input) before being written.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.perception.functional_groups import detect_functional_groups


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.fixture
def gated_namer(monkeypatch):
    """A namer with the SELF-01 OPSIN validity gate ON (production behaviour).

    ``tests/conftest.py`` has an autouse fixture that disables the gate for the
    whole suite so tests can assert RAW producer output.  Every fail-closed
    assertion in this module is about what actually SHIPS, so it must re-enable
    the gate -- otherwise the test asserts an ungated candidate string and
    proves nothing about production.  (Without this, the benzyl case asserts
    against the raw 'methylbenzene' candidate rather than the abstention.)
    """
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    return Orthonym()


# --------------------------------------------------------------------------
# 1. Perception: the four new functional groups are detected
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,fg", [
    ("C[As](=O)(O)O", "arsonic_acid"),
    ("CC[As](=O)(O)O", "arsonic_acid"),
    ("c1ccccc1[As](=O)(O)O", "arsonic_acid"),
    ("C[As](C)(=O)O", "arsinic_acid"),
    ("C[Sb](=O)(O)O", "stibonic_acid"),
    ("C[Sb](C)(=O)O", "stibinic_acid"),
])
def test_pnictogen_oxoacid_functional_group_detected(smiles, fg):
    mol = Chem.MolFromSmiles(smiles)
    assert fg in detect_functional_groups(mol), (
        f"{smiles} should perceive {fg}"
    )


# --------------------------------------------------------------------------
# 2. The carbon guard: a FREE inorganic oxoacid must NOT be claimed
#    (arsoric acid As(O)(OH)3 / stiboric acid are separate preselected names)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,fg", [
    ("O[As](=O)(O)O", "arsonic_acid"),    # arsoric acid -- no C-As bond
    ("O[Sb](=O)(O)O", "stibonic_acid"),   # stiboric acid -- no C-Sb bond
])
def test_free_inorganic_oxoacid_is_not_an_organo_oxoacid(smiles, fg):
    mol = Chem.MolFromSmiles(smiles)
    assert fg not in detect_functional_groups(mol), (
        f"{smiles} is a free inorganic oxoacid and must not match {fg}"
    )


# --------------------------------------------------------------------------
# 3. End-to-end PIN names (all OPSIN-verified against the input structure)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # -onic (one C, two OH) -- BB L36051 / L36054
    ("C[As](=O)(O)O",            "methylarsonic acid"),
    ("CC[As](=O)(O)O",           "ethylarsonic acid"),
    ("CCC[As](=O)(O)O",          "propylarsonic acid"),
    ("c1ccccc1[As](=O)(O)O",     "phenylarsonic acid"),
    ("C[Sb](=O)(O)O",            "methylstibonic acid"),
    ("CC[Sb](=O)(O)O",           "ethylstibonic acid"),
    # -inic (two C, one OH) -- BB L36052 / L36054
    ("C[As](C)(=O)O",            "dimethylarsinic acid"),
    ("c1ccccc1[As](=O)(O)c1ccccc1", "diphenylarsinic acid"),
    ("C[Sb](C)(=O)O",            "dimethylstibinic acid"),
    # BB L36066 verbatim PIN: C6H5-As(CH3)(O)OH
    ("c1ccccc1[As](C)(=O)O",     "methyl(phenyl)arsinic acid"),
])
def test_pnictogen_oxoacid_pin_name(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------
# 4. The phosphorus sibling is unchanged (this build generalises its code path)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CP(=O)(O)O",             "methylphosphonic acid"),
    ("CCP(=O)(O)O",            "ethylphosphonic acid"),
    ("CCCP(=O)(O)O",           "propylphosphonic acid"),
    ("OP(=O)(O)c1ccccc1",      "phenylphosphonic acid"),
    ("CP(C)(=O)O",             "dimethylphosphinic acid"),
    ("c1ccccc1P(C)(=O)O",      "methyl(phenyl)phosphinic acid"),
])
def test_phosphorus_oxoacid_names_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------
# 5. Fail-closed: a complex organyl defers (never a wrong name).
#    Symmetric with the phosphorus sibling -- benzyl is refused on BOTH.
#    (Lifting this is the DROP-24 keystone, deliberately out of scope here.)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "C1=CC=C(C=C1)C[As](=O)(O)O",   # benzylarsonic acid
    "C1=CC=C(C=C1)CP(=O)(O)O",      # benzylphosphonic acid
])
def test_complex_organyl_fails_closed(gated_namer, smiles):
    """Must never emit a WRONG name; abstention is the correct outcome."""
    from orthonym.errors import is_failure_name
    name = gated_namer.name(smiles)
    assert is_failure_name(name), (
        f"{smiles} must fail closed, got {name!r}"
    )


@pytest.mark.parametrize("rules_fn,fg,smiles", [
    ("name_arsonic_acid", "arsonic_acid", "C1=CC=C(C=C1)C[As](=O)(O)O"),
    ("name_arsonic_acid", "arsonic_acid", "CC(C)(C)[As](=O)(O)O"),
    ("name_stibonic_acid", "stibonic_acid", "C1=CC=C(C=C1)C[Sb](=O)(O)O"),
    ("name_arsinic_acid", "arsinic_acid", "C1=CC=C(C=C1)C[As](C)(=O)O"),
    ("name_stibinic_acid", "stibinic_acid", "C1=CC=C(C=C1)C[Sb](C)(=O)O"),
])
def test_producer_itself_fails_closed_on_complex_organyl(rules_fn, fg, smiles):
    """The NAMER must return None -- not merely be caught downstream.

    Asserting only the shipped string is too weak: the SELF-01 OPSIN gate would
    mask a producer that fabricates a wrong name (a mutation that made the
    namer emit 'methylarsonic acid' for benzylarsonic acid survived a
    shipped-output-only test).  The OPSIN gate also fails OPEN when no JRE is
    present, so producer-level fail-closed is the real safety property.
    """
    from orthonym.rules import phosphorus
    mol = Chem.MolFromSmiles(smiles)
    matches = detect_functional_groups(mol).get(fg, [])
    assert matches, f"{smiles} should still perceive {fg}"
    assert getattr(phosphorus, rules_fn)(mol, matches[0]) is None, (
        f"{rules_fn} must fail closed on {smiles}"
    )


@pytest.mark.parametrize("rules_fn,fg,smiles,wrong_token", [
    # benzyl (7 C, one in a ring) was counted as a LINEAR heptyl chain
    ("name_phosphinic_acid", "phosphinic_acid", "C1=CC=C(C=C1)CP(C)(=O)O", "heptyl"),
    ("name_arsinic_acid", "arsinic_acid", "C1=CC=C(C=C1)C[As](C)(=O)O", "heptyl"),
    ("name_stibinic_acid", "stibinic_acid", "C1=CC=C(C=C1)C[Sb](C)(=O)O", "heptyl"),
    # cyclohexyl (6 C, all in a ring) was counted as a LINEAR hexyl chain
    ("name_phosphinic_acid", "phosphinic_acid", "C1CCCCC1P(C)(=O)O", "hexyl"),
    ("name_arsinic_acid", "arsinic_acid", "C1CCCCC1[As](C)(=O)O", "hexyl"),
])
def test_cyclic_substituent_is_never_renamed_as_a_linear_chain(
    rules_fn, fg, smiles, wrong_token,
):
    """REGRESSION (pre-existing accuracy bug found in this phase).

    The -inic path called ``_characterize_substituent`` raw, which counts the
    carbons of a non-aromatic subtree as a linear chain.  So a cyclic
    substituent was silently renamed to a WRONG CONSTITUTION:
    benzyl(methyl)phosphinic acid -> 'heptyl(methyl)phosphinic acid',
    cyclohexyl(methyl) -> 'hexyl(methyl)'.  The -onic path never had this bug
    because it went through ``pure_organyl_prefix_name``; the fix routes -inic
    through the same gate for P, As and Sb alike.
    """
    from orthonym.rules import phosphorus
    mol = Chem.MolFromSmiles(smiles)
    matches = detect_functional_groups(mol).get(fg, [])
    assert matches, f"{smiles} should perceive {fg}"
    result = getattr(phosphorus, rules_fn)(mol, matches[0])
    assert result is None, (
        f"{rules_fn} must refuse the cyclic substituent in {smiles}, "
        f"got {result!r} (contains the fabricated '{wrong_token}')"
    )


@pytest.mark.parametrize("smiles", [
    "C1=CC=C(C=C1)C[As](=O)(O)O",
    "C1=CC=C(C=C1)C[Sb](=O)(O)O",
])
def test_benzyl_case_fails_closed_even_without_the_opsin_gate(monkeypatch, smiles):
    """Safety here must not depend on Java being installed.

    The OPSIN validity gate fails OPEN when no JAR is present, so a refusal that
    only happens inside the gate is not a real guarantee.  Before this phase the
    engine emitted the constitutionally-WRONG 'methylbenzene' for benzylarsonic
    acid and relied entirely on SELF-01 to suppress it; now the arsonic
    principal group is perceived and the producer refuses on its own.
    """
    import orthonym.namer as _namer
    from orthonym.errors import is_failure_name
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)
    name = Orthonym().name(smiles)
    assert is_failure_name(name), f"ungated output must still refuse, got {name!r}"
    assert "methylbenzene" not in name and "toluene" not in name


@pytest.mark.parametrize("smiles,element", [
    ("OCCP(=O)(O)O", "phosphon"),
    ("OCC[As](=O)(O)O", "arsonic"),
    ("OCC[Sb](=O)(O)O", "stibonic"),
])
def test_heteroatom_organyl_defers_to_the_generic_path_on_every_element(
    gated_namer, smiles, element,
):
    """A substituent the simple namer cannot prove defers, it does not abstain.

    ``pure_organyl_prefix_name`` refuses a heteroatom-bearing organyl, so the
    handler returns None and the GENERIC suffix assembler emits the
    parent-hydride-stem form (``2-hydroxyethane-1-phosphonic acid``).  That form
    is NOT the PIN -- the PIN is ``(2-hydroxyethyl)arsonic acid`` -- but it is
    structure-correct and OPSIN-exact, and it is pre-existing shipped behaviour
    on phosphorus.  This test pins that As/Sb behave the SAME as P here rather
    than diverging in either direction.
    """
    name = gated_namer.name(smiles)
    assert element in name, f"{smiles} -> {name!r}"


@pytest.mark.parametrize("smiles", [
    "CCP(=O)O",     # ethylphosphinic acid  (P sibling)
    "CC[As](=O)O",  # ethylarsinic acid     (As)
    "CC[Sb](=O)O",  # ethylstibinic acid    (Sb, BB L36064 PIN)
])
def test_monosubstituted_inic_acid_is_a_symmetric_shared_gap(gated_namer, smiles):
    """AUDITED PRE-EXISTING GAP, deliberately left symmetric.

    BB L36064 gives ``ethylstibinic acid`` as a PIN, so a MONO-substituted
    -inic acid (one C + one H on the central atom) is a real class member.
    The shipped phosphorus path has never covered it -- ``CCP(=O)O`` perceives
    no functional group and fails closed -- and this phase mirrors that scope
    rather than making As/Sb silently better than P.  Widening it must be done
    for all three elements at once.

    This test exists to PIN the symmetry: if any one element starts naming
    mono-substituted -inic acids, all three must.
    """
    from orthonym.errors import is_failure_name
    assert is_failure_name(gated_namer.name(smiles))


def test_complex_organyl_never_named_as_the_bare_parent(gated_namer):
    """The specific historical leak: dropping the ring and naming the rest."""
    name = gated_namer.name("C1=CC=C(C=C1)C[As](=O)(O)O")
    assert "methylbenzene" not in name
    assert "toluene" not in name


# --------------------------------------------------------------------------
# 6. Element routing: genuine organometallics and organo-pnictogen hydrides
#    are NOT stolen by the new acid path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("C[As](C)C",       "trimethylarsane"),
    ("C[Sb](C)C",       "trimethylstibane"),
    ("CC[As](CC)CC",    "triethylarsane"),
    ("Cl[As](Cl)Cl",    "trichloroarsane"),
    ("Br[Sb](Br)Br",    "tribromostibane"),
])
def test_organo_pnictogen_hydrides_still_named(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles", [
    "[cH-]1cccc1.[cH-]1cccc1.[Fe+2]",   # ferrocene
    "C[Li]",                            # methyllithium
    "[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]",  # Fe(CO)5
])
def test_genuine_organometallics_still_route_to_p69(smiles):
    """The ORGANOMETALLIC@50 predicate must still claim real organometallics."""
    from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
    mol = Chem.MolFromSmiles(smiles)
    entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
    assert entry.predicate(mol, smiles, Chem.CanonSmiles(smiles), None) is True


@pytest.mark.parametrize("smiles", [
    "C[As](=O)(O)O",
    "C[Sb](=O)(O)O",
    "C[As](C)(=O)O",
])
def test_pnictogen_oxoacids_are_not_organometallic(smiles):
    """An arsonic acid is a P-67 oxoacid, never a P-69 organometallic."""
    from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
    mol = Chem.MolFromSmiles(smiles)
    entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
    assert entry.predicate(mol, smiles, Chem.CanonSmiles(smiles), None) is False


# --------------------------------------------------------------------------
# 7. Honesty of the descriptive fallback: a carbon-bearing arsenic compound
#    is not "inorganic"
# --------------------------------------------------------------------------

def test_organo_arsenic_failure_is_not_labelled_inorganic(gated_namer):
    """C7H9AsO3 bearing a benzyl group is an ORGANO-arsenic compound."""
    name = gated_namer.name("C1=CC=C(C=C1)C[As](=O)(O)O")
    assert "inorganic" not in name, (
        f"a carbon-bearing arsenic compound must not be called inorganic: {name!r}"
    )
    assert "arsenic" in name


def test_carbon_free_antimony_fallback_label_is_preserved():
    """The no-carbon case is genuinely inorganic -- canary must not move."""
    from orthonym.errors import classify_failure_limit
    mol = Chem.MolFromSmiles("[O]=[Sb]([O-])([O-])[OH]")
    assert classify_failure_limit(mol).message == "antimony compound (not supported)"
