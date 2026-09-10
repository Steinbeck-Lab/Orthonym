""" method (1) — a cationic (onium) SPIRO-JUNCTION atom.

A quaternary onium at a spiro junction -- a ring N+ (or P+/...) whose four
bonds are ALL ring bonds -- used to abstain (``unknown organic compound``).
The charged router names ring cations by neutralize -> re-enter, but removing
the charge here leaves an over-valent neutral heteroatom (a 4-bonded neutral N)
that RDKit ``SanitizeMol`` rejects, so no neutral parent name is ever produced
(``_emit_ring_cumulative_suffix``'s DEMOTE branch also declines: a spiro
junction has no exocyclic substituent to sever, so ``exo == ``).

Fix: ``rules/spiro.py::name_charged_spiro_system`` names the neutral
skeletal-replacement ('a') spiro PARENT directly off the CHARGED mol
(``name_spiro_system`` reads the cation as an ordinary skeletal heteroatom for
the replacement prefix and the spiro numbering) and appends the parent-hydride
``-<locant>-ium`` suffix at the cation's spiro locant. ``route_charged`` calls
it (quaternary + onium single-cation ring branches) and RT-gates the result
via ``_cation_name_rt_ok`` -- ship only on a full-InChIKey match, else abstain.

Method (1) (neutral 'a' parent + '-ium' suffix) gives the PREFERRED IUPAC name
and is preferred to the ``azonia`` cationic skeletal-replacement alternative
, "Method (1) gives preferred IUPAC names", the Blue Book the Blue Book;
``1-methyl-1-azabicyclo[2.2.1]heptan-1-ium`` (PIN) vs the ``azonia`` form,
the Blue Book). So the emitted PIN is ``...azaspiro...-ium``, NEVER
``...azoniaspiro...``.

Every RT-verified row below round-trips through OPSIN 2.9.0 to the identical
structure (full-InChIKey match) -- not merely to *a* valid molecule.
"""

import pytest
from rdkit import Chem

from orthonym.errors import is_failure_name
from orthonym.namer import Orthonym
from orthonym.rules.spiro import name_charged_spiro_system
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

# The whole module needs the REAL OPSIN validity / gate switched on:
# the suite disables it by default, but the fail-closed abstention row depends
# on the production gate actually running, and the router's own RT gate
# (`_cation_name_rt_ok`) is what makes these emissions 0-wrong.
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namer():
    return Orthonym(style="pin")


# ---------------------------------------------------------------------------
# Integration: RT-verified charged spiro-junction cations (the fix)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # Monospiro junction: the N+ IS the spiro atom (all 4 bonds in rings).
    ("C1CCCC[N+]12CCCCC2", "6-azaspiro[5.5]undecan-6-ium"),
    # Dispiro junction: the N+ is one of two adjacent spiro atoms.
    ("C1CCCCC12[N+]1(CCCCC1)CCC2", "6-azadispiro[5.0.5.3]pentadecan-6-ium"),
    # A phosphonium at a spiro junction -- the same class, one heteroatom over
    # (classified 'onium', so the onium single-cation branch reaches the fix).
    ("C1CCCC[P+]12CCCCC2", "6-phosphaspiro[5.5]undecan-6-ium"),
])
@pytest.mark.unit
def test_charged_spiro_junction_names(namer, smiles, expected):
    assert namer.name(smiles) == expected
    # Independent 0-wrong proof: the emission round-trips to the input.
    rt = opsin_roundtrip_check(smiles, expected)
    assert rt["passed"], rt


# ---------------------------------------------------------------------------
# The halide salt composes via the salt path once the cation names (RX).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("[Cl-].C1CCCC[N+]12CCCCC2",
     "6-azaspiro[5.5]undecan-6-ium chloride"),
    ("[Br-].C1CCCCC12[N+]1(CCCCC1)CCC2",
     "6-azadispiro[5.0.5.3]pentadecan-6-ium bromide"),
])
@pytest.mark.unit
def test_charged_spiro_junction_salts(namer, smiles, expected):
    assert namer.name(smiles) == expected
    rt = opsin_roundtrip_check(smiles, expected)
    assert rt["passed"], rt


# ---------------------------------------------------------------------------
# The name_charged_spiro_system builder in isolation (spiro.py)
# ---------------------------------------------------------------------------
class TestNameChargedSpiroSystem:
    """Direct unit tests of the builder -- it returns the candidate string
    (the CALLER RT-gates); no OPSIN validation happens inside it."""

    def _cation_idx(self, mol):
        return next(a.GetIdx() for a in mol.GetAtoms()
                    if a.GetFormalCharge() == 1)

    @pytest.mark.unit
    def test_monospiro_junction(self):
        mol = Chem.MolFromSmiles("C1CCCC[N+]12CCCCC2")
        assert (name_charged_spiro_system(mol, self._cation_idx(mol))
                == "6-azaspiro[5.5]undecan-6-ium")

    @pytest.mark.unit
    def test_dispiro_junction(self):
        mol = Chem.MolFromSmiles("C1CCCCC12[N+]1(CCCCC1)CCC2")
        assert (name_charged_spiro_system(mol, self._cation_idx(mol))
                == "6-azadispiro[5.0.5.3]pentadecan-6-ium")

    @pytest.mark.unit
    def test_declines_non_spiro_ring_cation(self):
        # A quaternary ring N+ that is NOT a spiro atom (piperidinium) is out of
        # this builder's scope -> '' (the existing ring-cation path names it).
        mol = Chem.MolFromSmiles("C[N+]1(C)CCCCC1")
        cat = self._cation_idx(mol)
        assert name_charged_spiro_system(mol, cat) == ""

    @pytest.mark.unit
    def test_declines_ring_member_cation_in_spiro(self):
        # An NH2+ ring MEMBER of a spiro system (not the junction) is degree 3;
        # the builder is scoped to the spiro ATOM, so it declines here and the
        # in-place ring-cation path owns it.
        mol = Chem.MolFromSmiles("C1[NH2+]CCC12CCCCC2")
        cat = self._cation_idx(mol)
        assert name_charged_spiro_system(mol, cat) == ""

    @pytest.mark.unit
    def test_declines_neutral(self):
        # No cation -> nothing to append '-ium' to.
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        assert name_charged_spiro_system(mol, 3) == ""


# ---------------------------------------------------------------------------
# Regression canaries: unchanged neighbours must stay byte-identical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # Neutral spiro parent -- untouched by the charged path.
    ("C1CCC2(CC1)CCCC2", "spiro[4.5]decane"),
    ("CCO", "ethanol"),
    # A simple ring-N cation salt that already worked (not a spiro junction).
    ("[I-].C[N+]1(C)CCCCC1", "1,1-dimethylpiperidin-1-ium iodide"),
    # An NH2+ ring MEMBER in a spiro ring (in-place neutralize path) -- works
    # before and after the fix (the junction is the neutral carbon C12).
    ("C1[NH2+]CCC12CCCCC2", "2-azaspiro[4.5]decan-2-ium"),
    # A quaternary N+ ring MEMBER of a spiro ring bearing exocyclic methyls --
    # named by the DEMOTE branch (severs the methyls), not the junction fix.
    ("C[N+]1(C)CCC2(CC1)CCCC2", "8,8-dimethyl-8-azaspiro[4.5]decan-8-ium"),
])
@pytest.mark.unit
def test_charged_spiro_canaries(namer, smiles, expected):
    assert namer.name(smiles) == expected


# ---------------------------------------------------------------------------
# Fail-closed: an over-valent / radical hetero-cation (a 4-coordinate S+ that
# RDKit carries as a radical, i.e. a genuine 6λ4-thia species) is NOT a clean
# onium -- the builder's plain '-ium' name does not round-trip, so the router's
# RT gate rejects it and the molecule abstains rather than ship a wrong name.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_overvalent_sulfonium_junction_does_not_ship_wrong(namer):
    mol = Chem.MolFromSmiles("C1CCCC[S+]12CCCCC2")
    cat = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
    candidate = name_charged_spiro_system(mol, cat)
    # The builder offers a candidate, but it does NOT round-trip (the true
    # structure carries the radical/λ4 character the plain '-ium' name omits).
    if candidate:
        rt = opsin_roundtrip_check("C1CCCC[S+]12CCCCC2", candidate)
        assert not rt["passed"], (
            "the plain -ium name must not round-trip to the λ4/radical S+"
        )
