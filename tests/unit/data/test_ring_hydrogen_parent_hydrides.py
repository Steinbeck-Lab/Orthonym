"""Parent hydrides that need indicated hydrogen cite it, at both tiers: the Group-15
retained parents of /, and Hantzsch-Widman monocycles with
two indicated hydrogen atoms.

- (the Blue Book) "in a preferred IUPAC name a locant and the symbol 'H' must
  be cited"; (:14607) "all indicated hydrogen atoms must be cited";
  (:24639) "indicated hydrogen must always be cited when present in the corresponding
  structure". Table 2.9 lists 'indole (PIN)' / 'phosphindole (PIN)' side by side (:11708)
  and entry (19) says "the PIN is 1H-indole" (:11605). '10H-phenoxaphosphinine'
  for "phenoxaphosphinine (PIN, 10H-isomer shown)" (:11801), as "the PIN is
  10H-phenoxazine" (:11781). The table spellings are listing spellings, by analogy with
   Note 1 (:14685, the note of the seniority lists: "Indicated hydrogen atoms
  are not shown in this kind of listing").
- (:8320) every atom left without a double bond "is designated by indicated
  hydrogen. If there is a choice, such ring atoms are assigned low locants": '2H,4H-1,3-
  dioxine' (the set {2,4} is lower than {2,6}).
Every expected name reads back to the input's full InChIKey with OPSIN 2.9.0."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

ROWS = [
    ("C1=Cc2ccccc2[AsH]1", "1H-arsindole"),                      #:11708
    ("C1=c2ccccc2=C[AsH]1", "2H-isoarsindole"),                  #:11710
    ("c1ccc2[pH]ccc2c1", "1H-phosphindole"),                     #:11708
    ("c1ccc2c[pH]cc2c1", "2H-isophosphindole"),                  #:11710
    ("c1ccc2c(c1)Oc1ccccc1P2", "10H-phenoxaphosphinine"),        #:11801
    ("c1ccc2c(c1)Oc1ccccc1[AsH]2", "10H-phenoxarsinine"),        #:11805
    ("c1cc[c]2c(c1)Oc1cccc[c]1[SbH]2", "10H-phenoxastibinine"),  #:11807
    ("c1ccc2c(c1)Sc1ccccc1[AsH]2", "10H-phenothiarsinine"),      #:11811
    ("C1=COCOC1", "2H,4H-1,3-dioxine"),
    ("CC1=COCOC1", "5-methyl-2H,4H-1,3-dioxine"),
    ("OC(=O)C1=COCOC1", "2H,4H-1,3-dioxine-5-carboxylic acid"),
]

#: one indicated hydrogen: unchanged
UNCHANGED = [
    ("C1=COCO1", "2H-1,3-dioxole"),
    ("C1=CCC=CO1", "4H-pyran"),
    ("c1ccc2c(c1)Oc1ccccc1S2", "phenoxathiine"),                 #:11795, no hydrogen
    ("C1=CC2=CC=C[As]2C=C1", "arsindolizine"),                   #:11709, none needed
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="ring-hydrogen"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ROWS + UNCHANGED)
def test_a_parent_hydride_cites_its_indicated_hydrogen(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


#: a substituted ring whose parent has two indicated hydrogen atoms is numbered by them
#: before its prefixes: (the Blue Book-25044) (c) indicated hydrogen before
#: (g) substituents named as prefixes, and before (d) a suffix that needs no hydrogen
#: ('2H-pyran-6-carboxylic acid (PIN)',:3252). '6-methyl-2H,4H-1,3-dioxine', not
#: '4-methyl-2H,6H-1,3-dioxine' (the same molecule, OPSIN 2.9.0 FULL for both) and not
#: '4-methyl-1,3-dioxine' (another molecule). The default tier declined these rows and
#: best-effort wrote a replacement name ('4-methyl-1,3-dioxacyclohex-4-ene').
SUBSTITUTED_TWO_INDICATED = [
    ("CC1=CCOCO1", "6-methyl-2H,4H-1,3-dioxine"),
    ("CC1=CCSCS1", "6-methyl-2H,4H-1,3-dithiine"),
    ("CC1=CCOC(C)O1", "2,6-dimethyl-2H,4H-1,3-dioxine"),
    ("OC(=O)C1=CCOCO1", "2H,4H-1,3-dioxine-6-carboxylic acid"),
    ("NC1=CCOCO1", "2H,4H-1,3-dioxin-6-amine"),
    ("CC1OCOC=C1", "4-methyl-2H,4H-1,3-dioxine"),                # unchanged
    ("CC1=CCOCS1", "4-methyl-2H,6H-1,3-oxathiine"),              # unchanged: O1, S3 fix it
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", SUBSTITUTED_TWO_INDICATED)
def test_a_substituted_ring_is_numbered_by_its_indicated_hydrogen(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
