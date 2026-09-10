""" abstain-recovery — salt / charged multi-fragment component coverage.

A salt (``cation anion`` binary name, P-77.1.1) abstained whenever ONE
component was named WRONG by the ordinary per-component namer -- even though the
same component names correctly standalone via the full engine. Measured root
cause: ``name_cation`` returns a CONFIDENTLY-WRONG name for a complex organic
cation (a fused azatetracyclo cage cation -> ``tropan-1-ium``, a different,
smaller ring system; the same cation is also named ``...heptaen-3-onium`` in a
nested naming context). Being non-empty, that wrong name shadowed the
best-effort per-component fallback (``if not name:`` never fired), so only the
top-level SELF-01 joined-salt gate caught it and the WHOLE salt abstained.

Fix (``rules/salts.py``): under a best-effort flag, verify each ORGANIC ion
component round-trips to its own fragment (``_ion_fragment_roundtrips``); when
the ordinary name does not round-trip, replace it with the best-effort
per-component name (which the fresh instance's own gate already RT-verifies).
Replace ONLY when best-effort yields a verified alternative, so a correct
ordinary name is never discarded -- the PIN tier and every canary salt stay
byte-identical. Metal / inorganic table words (sodium, chloride,...) are
untouched (curated single source of truth, covered by the joined gate).

0-WRONG ABSOLUTE: every component round-trips AND the whole salt OPSIN-round-
trips to the full multi-fragment InChIKey, else the salt abstains. Governing
rules: P-77.1.1 (binary salt name), P-73.1 (-ium cation suffix), P-23.2
(von Baeyer cage parent).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym, name_compound
from orthonym.rules.salts import _ion_fragment_roundtrips

# Best-effort tier flags -- the recovery is scoped to these (PIN tier is
# byte-identical because all three are False there).
FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
             allow_aromatic_general=True)


def _ik(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return inchi.MolToInchiKey(m) if m is not None else None


def _full_rt_matches(name: str, input_smiles: str) -> bool:
    """True iff *name* OPSIN-parses back to the FULL multi-fragment InChIKey of
    *input_smiles* (the project's headline round-trip identity)."""
    if not name or name == "unknown organic compound":
        return False
    from orthonym.namer import _validity_gate_name_to_smiles
    smi = _validity_gate_name_to_smiles(name)
    return bool(smi) and _ik(smi) == _ik(input_smiles)


@pytest.fixture(scope="module")
def be():
    return Orthonym(**FLAGS)


# ---------------------------------------------------------------------------
# The recovery -- two structurally distinct cage/complex-cation salts.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
class TestCageCationSaltRecovery:
    def test_azatetracyclo_cage_cation_chloride(self, be):
        # A bridged fused azatetracyclo cage cation + chloride. The bare cation
        # names fine standalone; the salt abstained because name_cation returned
        # the wrong 'tropan-1-ium'. Now RT-replaced with the true cage name.
        smi = "C[NH+]1C2CCC1C3CC4=CC=CC=C4CC3C2.[Cl-]"
        name = be.name(smi)
        assert name == (
            "16-methyl-16-azatetracyclo[11.2.1.0^3,12.0^5,10]"
            "hexadeca-5,7,9-trien-16-ium chloride"
        )
        assert _full_rt_matches(name, smi)

    def test_triaza_tetracyclo_cation_chloride(self, be):
        # A fused triaza-tetracyclo (fused-heteroaromatic) cage cation + chloride.
        smi = "C1C2=CC=CC=C2C3=NC(=O)C4=C([NH+]31)N=CC=C4.[Cl-]"
        name = be.name(smi)
        assert name == (
            "3-oxo-2,8,10-triazatetracyclo[8.7.0.0^4,9.0^12,17]"
            "heptadeca-1,4,6,8,12,14,16-heptaen-10-ium chloride"
        )
        assert _full_rt_matches(name, smi)

    def test_diaryliodanium_triflate(self, be):
        # Diaryliodanium (hypervalent-I cation, Task 1) triflate -- the salt
        # router composes it once each component is nameable. Canary + recovery.
        smi = "c1ccccc1[I+]c1ccccc1.[O-]S(=O)(=O)C(F)(F)F"
        name = be.name(smi)
        assert name == "diphenyliodanium trifluoromethanesulfonate"
        assert _full_rt_matches(name, smi)


# ---------------------------------------------------------------------------
# The per-component 0-wrong oracle -- accepts a correct ion word, rejects a
# confidently-wrong one. This is what lets a wrong ordinary name be replaced.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
class TestIonFragmentRoundtrips:
    def test_correct_cage_cation_name_roundtrips(self):
        cat = Chem.MolFromSmiles("C[NH+]1C2CCC1C1Cc3ccccc3CC1C2")
        good = ("16-methyl-16-azatetracyclo[11.2.1.0^3,12.0^5,10]"
                "hexadeca-5,7,9-trien-16-ium")
        assert _ion_fragment_roundtrips(cat, good) is True

    def test_wrong_ordinary_cation_name_is_rejected(self):
        # 'tropan-1-ium' is a different, smaller ring system -- must NOT be
        # accepted for the cage cation.
        cat = Chem.MolFromSmiles("C[NH+]1C2CCC1C1Cc3ccccc3CC1C2")
        assert _ion_fragment_roundtrips(cat, "tropan-1-ium") is False

    def test_empty_and_none_fail_closed(self):
        cat = Chem.MolFromSmiles("C[NH3+]")
        assert _ion_fragment_roundtrips(cat, "") is False
        assert _ion_fragment_roundtrips(None, "methanaminium") is False


# ---------------------------------------------------------------------------
# Canaries -- must stay byte-identical (PIN tier + best-effort tier).
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
class TestCanariesByteIdentical:
    @pytest.mark.parametrize("smi,expected", [
        ("[Na+].CC(=O)[O-]", "sodium acetate"),
        ("CC(=O)[O-].C[NH3+]", "methanaminium acetate"),
        ("C[NH3+].[Cl-]", "methanaminium chloride"),
        # P-73.4 verbatim (PIN) azabicyclo-ium salt.
        ("C[N+]12CCC(CC1)C2.[Cl-]",
         "1-methyl-1-azabicyclo[2.2.1]heptan-1-ium chloride"),
        ("O=C([O-])CCC(=O)O.[NH4+]", "ammonium 3-carboxypropanoate"),
    ])
    def test_salt_canary_best_effort(self, be, smi, expected):
        assert be.name(smi) == expected

    @pytest.mark.parametrize("smi,expected", [
        ("[Na+].CC(=O)[O-]", "sodium acetate"),
        ("CC(=O)[O-].C[NH3+]", "methanaminium acetate"),
        ("CCO", "ethanol"),
        ("OC(=O)C(=O)O", "oxalic acid"),
    ])
    def test_pin_tier_byte_identical(self, smi, expected):
        assert name_compound(smi, style="pin") == expected

    def test_neutral_adduct_unaffected(self, be):
        # Neutral multi-fragment goes through the adduct namer (not name_salt);
        # the fix must not touch it.
        assert be.name("O.OC(=O)C(=O)O") == "oxalic acid—water (1/1)"
