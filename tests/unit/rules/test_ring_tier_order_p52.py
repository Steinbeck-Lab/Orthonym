"""Lock the ring-naming tier order, with von Baeyer as the RESIDUAL tier.

Order under test: retained/catalog -> PIN systematic -> fusion constructor
                   -> von Baeyer polyene LAST

BLUE BOOK AUTHORITY
-------------------
* **** (heading the Blue Book, sentence the Blue Book) -- "Fusion nomenclature
  gives preferred IUPAC names only to compounds having at least two rings of at
  least five or more members. This requirement is not necessarily applied in
  general nomenclature, in which names such as cyclopropabenzene and
  cyclobutabenzene can be used. *When fusion names are not allowed, unsaturated
  von Baeyer ring system names are preferred IUPAC names* (see."
  So von Baeyer is what you fall back TO, not what you reach for.
* **** (the Blue Book, list the Blue Book-19542) -- the senior polycyclic ring
  system occurs first in: (a) spiro, (b) cyclic phane, (c) fused, (d) bridged
  fused, (e) nonfused bridged ring system, (f) linear phane, (g) ring assembly.
  von Baeyer is item (e), 5th of 7 -- below fused. The list body says "nonfused
  bridged ring system"; the equivalence to von Baeyer is fixed by the subsection
  heading at the Blue Book and the contents row at the Blue Book.
* Structurally, has PIN-selection subsections for fusion, phane, fullerene
  and ring assemblies, and **none for von Baeyer** -- consistent with it being
  residual rather than selected.

CAVEAT deliberately encoded below: is NOT the top-level tiebreaker.
the Blue Book and the Blue Book both state that the skeletal-atom-count criterion
"supersedes, which prefers a fused ring to a bridged fused ring", so
the criteria run FIRST. This test therefore locks the tier order for
ring-NAMING of a single ring system, and does not assert the 7-class list as a
parent-selection tiebreaker.

THE TRAP THIS TEST EXISTS TO CATCH
----------------------------------
adamantane and cubane are retained AND are PINs, so they MUST keep their retained
names. quinuclidine and prismane are retained for general nomenclature only / no
longer recommended, so they MUST NOT. All four live in the same Blue Book table
(Table 2.6,, so a change that treats "retained von Baeyer name" as one
undifferentiated bucket breaks two of the four in one direction or the other.
"""

import os
import re

# Same pattern as tests/unit/assembly/test_serializer_flip_tripwire.py: keep the
# fast unit tier free of the OPSIN subprocess...
#... and put the environment back right after the import. Under xdist every
# worker imports every test module at collection, so a module-level write that
# stayed in os.environ switched the gates off in every child process the rest of
# the suite spawned (TRIAGE C1: subprocess tests saw gate-off names). The namer
# reads these variables once, at import, so restoring them changes nothing here.
_ENV_BEFORE = {k: os.environ.get(k) for k in (
    "ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", "ORTHONYM_SELF_CONSISTENCY_GATE")}
os.environ.setdefault("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", "1")
os.environ.setdefault("ORTHONYM_SELF_CONSISTENCY_GATE", "off")

import pytest
from rdkit import Chem

from orthonym import name_compound

for _k, _v in _ENV_BEFORE.items():   # restore: see the note above the import
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v

#: A von Baeyer descriptor: "bicyclo[", "tetracyclo[",... but NOT "cyclohexane"
#: and NOT "spiro[".
VON_BAEYER = re.compile(
    r"\b(?:bi|tri|tetra|penta|hexa|hepta|octa|nona|deca)cyclo\[", re.I
)

# --- one representative per tier, so the ORDER itself is exercised ------------
CUBANE = "C12C3C4C1C1C2C3C41"            # retained AND a PIN (the Blue Book)
ADAMANTANE = "C1C2CC3CC1CC(C2)C3"        # retained AND a PIN (the Blue Book)
QUINUCLIDINE = "C1CN2CCC1CC2"            # retained, general only (the Blue Book)
PRISMANE = "C12C3C1C1C3C21"              # no longer recommended (the Blue Book)


def name(smiles):
    return name_compound(Chem.CanonSmiles(smiles))


@pytest.mark.unit
class TestTier1RetainedNamesThatArePins:
    """Tier 1: a retained name that IS the PIN wins outright."""

    @pytest.mark.parametrize("smiles,expected", [
        (ADAMANTANE, "adamantane"),
        (CUBANE, "cubane"),
    ])
    def test_retained_pin_beats_von_baeyer(self, smiles, expected):
        """the Blue Book -- 'adamantane and cubane are used in general nomenclature
        AND as preferred IUPAC names'."""
        got = name(smiles)
        assert got == expected, (
            f"{expected} is a retained PIN (BB:9881) and must not be "
            f"systematised; got {got!r}"
        )
        assert not VON_BAEYER.search(got), (
            f"{expected} must not fall through to von Baeyer; got {got!r}"
        )


@pytest.mark.unit
class TestTier2NonPinRetainedNamesAreOverridden:
    """Tier 2: when the retained name is NOT the PIN, the systematic name wins.

    This is the other half of the trap: same Blue Book table as tier 1.
    """

    def test_quinuclidine_is_systematised(self):
        """the Blue Book 'retained for general nomenclature only';
        the Blue Book 'quinuclidine 1-azabicyclo[2.2.2]octane (PIN)'."""
        got = name(QUINUCLIDINE)
        assert got == "1-azabicyclo[2.2.2]octane", got
        assert "quinuclidin" not in got.lower()

    def test_prismane_is_systematised(self):
        """the Blue Book 'The name prismane is no longer recommended.'
        the Blue Book gives the systematic name (superscripts OCR-flattened on disk;
        the correct form is tetracyclo[2.2.0.0^2,6.0^3,5]hexane)."""
        got = name(PRISMANE)
        assert "prismane" not in got.lower(), (
            f"prismane is no longer recommended (BB:9881); got {got!r}"
        )
        assert VON_BAEYER.search(got), (
            f"prismane must fall through to the von Baeyer residual; got {got!r}"
        )
        assert got.startswith("tetracyclo["), got

    def test_prismane_reference_structure_is_really_prismane(self):
        """Guard the guard: 2 triangles + 3 squares, C6H6, all vertices degree 3."""
        mol = Chem.MolFromSmiles(PRISMANE)
        assert mol.GetNumAtoms() == 6
        assert all(a.GetDegree() == 3 for a in mol.GetAtoms())
        ring_sizes = sorted(len(r) for r in Chem.GetSymmSSSR(mol))
        assert ring_sizes.count(3) == 2, ring_sizes
        assert ring_sizes.count(4) == 3, ring_sizes


@pytest.mark.unit
class TestTier3FusionOutranksVonBaeyer:
    """Tier 3: a fusion-nameable system must NEVER receive a von Baeyer name.

     ranks fused (c) above nonfused bridged / von Baeyer (e).
    """

    @pytest.mark.parametrize("smiles,label", [
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("c1ccc2cc3ccccc3cc2c1", "anthracene"),
        ("c1ccc2ncccc2c1", "quinoline"),
        ("c1ccc2[nH]ccc2c1", "indole"),
        ("c1ccc2nccnc2c1", "quinoxaline"),
        ("c1ccc2ncncc2c1", "quinazoline"),
    ])
    def test_fusion_nameable_never_gets_von_baeyer(self, smiles, label):
        got = name(smiles)
        assert not VON_BAEYER.search(got), (
            f"{label} is fusion-nameable (two rings of >=5 members, "
            f"P-52.2.4.1) so von Baeyer must not fire; got {got!r}"
        )
        assert got.lower() != "unknown organic compound", (
            f"{label} must still be nameable; got {got!r}"
        )


@pytest.mark.unit
class TestTier4VonBaeyerIsTheResidual:
    """Tier 4: von Baeyer fires exactly when nothing above it may."""

    def test_plain_bridged_cage_with_no_retained_name_gets_von_baeyer(self):
        """bicyclo[2.2.2]octane has no retained name and no fusion name."""
        got = name("C1CC2CCC1CC2")
        assert got == "bicyclo[2.2.2]octane", got

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN GAP, P-52.2.4.1 (BB:23710): fusion names are not allowed when a "
        "ring has fewer than five members, and 'When fusion names are not "
        "allowed, unsaturated von Baeyer ring system names are preferred IUPAC "
        "names'. BB:23722 gives cyclobutabenzene -> bicyclo[4.2.0]octa-"
        "1,3,5,7-tetraene (PIN). Orthonym currently emits the fusion-style "
        "general name 'benzocyclobutene' instead, i.e. it applies fusion "
        "nomenclature where the Blue Book forbids it. strict=True so this fails "
        "loudly the moment the residual is wired up correctly."
    ))
    def test_four_membered_fusion_must_yield_von_baeyer_pin(self):
        got = name("C1Cc2ccccc21")
        assert got == "bicyclo[4.2.0]octa-1,3,5,7-tetraene", got

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN GAP, same rule P-52.2.4.1. BB:23718 gives 1H-cyclopropabenzene "
        "-> bicyclo[4.1.0]hepta-1,3,5-triene (PIN). Orthonym currently fails "
        "closed ('unknown organic compound'), which is safe but not the PIN."
    ))
    def test_three_membered_fusion_must_yield_von_baeyer_pin(self):
        got = name("C1c2ccccc21")
        assert got == "bicyclo[4.1.0]hepta-1,3,5-triene", got


@pytest.mark.unit
def test_tier_order_is_strict_across_all_four_representatives():
    """The four tiers resolve at four different tiers -- the order, end to end."""
    assert name(CUBANE) == "cubane"                              # tier 1
    assert name(QUINUCLIDINE) == "1-azabicyclo[2.2.2]octane"     # tier 2
    assert not VON_BAEYER.search(name("c1ccc2ccccc2c1"))         # tier 3
    assert name(PRISMANE).startswith("tetracyclo[")              # tier 4
