"""A molecule the PIN-only tier names ``pin_verified`` gets the same name and the same label at the default tier (lane TI2).

The default is the ``best-effort`` tier, whose producers run in front of, and widen, the producers of the PIN-only tier
(``emit_tier="pin"``). On 45 of 3,074 rows of three measured sets the PIN-only tier returned a ``pin_verified`` name
and the default tier another name or a lower label. Each class below is one cause, fixed at its own site; the
molecules are those of the measured rows and the expectations are the PIN-only tier's own name and label, read here at
both tiers (the invariant), and the name of the PIN-only tier (so that a change of the PIN-only tier is seen).

A cause is one test case id:

* ``lactone`` the alcohol component of an ester holds a lactone: the ring oxygen of a lactone is not "another
                   ester of a different anion" "When anions are different, two methods are used",
                   the Blue Book, method (2)), so the ester route of the default tier no longer labels the name
                   below the PIN (``rules.esters``);
* ``declined`` a decomposition that names nothing leaves the provenance of the molecule as it found it, so the
                   rescue of a fragment it declined does not label the name of the route that ships
                   (``decomposition.engine.try_decompose``);
* ``amido`` the located functional-group namer of the wider tier spelled an N-acyl group as the compound prefix
                   '(benzoylamino)', method (2) of "Substituents of the types -NH-CO-R and -NH-SO2-R"
                   (the Blue Book: "Method (1) generates preferred IUPAC names"), where the default tier's namer gives the amido prefix 'benzamido' (``assembly.
                   substituent_naming._located_fg_hetero_root``);
* ``disulfanyl`` '-S-S-R' is the prefix 'R-disulfanyl' "Peroxides, disulfides, diselenides, and
                   ditellurides", the Blue Book, '(methyldisulfanyl)methane (PIN)':27874), not the sulfanyl of a
                   sulfanyl the wider tier's ring-branch composer made of it (``assembly.substituent_enumerator._name_thio_ring_branch``);
* ``ring-prefix`` a ring the default tier's cascade declines and its promotion re-run names with the strict producer
                   ('adamantan-2-yl') is named with it at the wider tier too, so the name is not labelled as a ring prefix
                   only a general tier builds (``assembly.strict_ring_prefix.general_tier_ring_prefix``);
* ``organometallic`` the first handler of the dispatch table is asked the default tier's way, and where it declines an
                   organyl ligand and the molecule has a principal characteristic group (a ketone, an acid, an ester) the
                   wider reading of the ligand is offered after the other routes (``assembly.default_tier_offer``): the
                   classes expressed by suffixes are senior to a parent hydride SENIORITY ORDER FOR CLASSES: the
                   classes 1 to 20 come before the classes based on the senior atom, 21 to 43), so a silyl ether of such
                   a molecule is named on its ring or chain.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: (cause, SMILES, the name and label of the PIN-only tier)
CASES = [
    ("lactone", "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCC(=O)Oc1ccc2ccc(=O)oc2c1",
     "2-oxo-2H-1-benzopyran-7-yl (6Z,9Z,12Z)-octadeca-6,9,12-trienoate"),
    ("lactone", "C/C=C/c1oc(=O)c(CO)c(OC)c1COC(C)=O",
     "{3-(hydroxymethyl)-4-methoxy-2-oxo-6-[(1E)-prop-1-en-1-yl]-2H-pyran-5-yl}methyl acetate"),
    ("lactone", "COc1cc(OC)c2oc(=O)c(COC(C)=O)cc2c1",
     "(6,8-dimethoxy-2-oxo-2H-1-benzopyran-3-yl)methyl acetate"),
    ("declined", "CC(Oc1cc(/C=C2\\SC(=S)N(CC(=O)O)C2=O)ccc1OCCc1ccccc1)c1ccccc1",
     "[(5Z)-4-oxo-5-{[3-(1-phenylethoxy)-4-(2-phenylethoxy)phenyl]methylidene}-2-sulfanylidene-"
     "1,3-thiazolidin-3-yl]acetic acid"),
    ("amido", "CC(=O)OC[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)c1ccccc1",
     "(2S)-2-[(2S)-2-benzamido-3-phenylpropanamido]-3-phenylpropyl acetate"),
    ("amido", "CC(O)C(NC(=O)C1CCCN1C)C1OC(SCCOC(=O)c2ccccc2N)C(O)C(O)C1O",
     "2-({3,4,5-trihydroxy-6-[2-hydroxy-1-(1-methylpyrrolidine-2-carboxamido)propyl]oxan-2-yl}sulfanyl)ethyl "
     "2-aminobenzoate"),
    ("disulfanyl", "Cc1occc1SSCc1ccco1", "3-{[(furan-2-yl)methyl]disulfanyl}-2-methylfuran"),
    ("ring-prefix", "CCOCC(C1C2CC3CC(C2)CC1C3)C(=O)OCC",
     "ethyl 2-(adamantan-2-yl)-3-ethoxypropanoate"),
    ("organometallic", "C[Si](C)(C)OC12CCC(CC1)C(=O)C2",
     "4-[(trimethylsilyl)oxy]bicyclo[2.2.2]octan-2-one"),
    ("organometallic", "C[C@]1(CCC(=O)C=C1C=C)CO[Si](C)(C)C(C)(C)C",
     "(4R)-4-({[tert-butyldi(methyl)silyl]oxy}methyl)-3-ethenyl-4-methylcyclohex-2-en-1-one"),
    ("organometallic", "CC(C)(C)[Si](C)(C)OCC1=CC=C(SS1)C=O",
     "6-({[tert-butyldi(methyl)silyl]oxy}methyl)-1,2-dithiine-3-carbaldehyde"),
    ("organometallic", "CCCCCC(CCC1=CC[C@@H]([C@@H]1CCCCCCC(=O)O)O)O[Si](C)(C)C(C)(C)C",
     "7-[(1R,5S)-2-(3-{[tert-butyldi(methyl)silyl]oxy}octyl)-5-hydroxycyclopent-2-en-1-yl]heptanoic acid"),
    ("organometallic", "CC(C)(C)OC(=O)C(CO[Si](C)(C)C(C)(C)C)C(=O)O",
     "3-tert-butoxy-2-({[tert-butyldi(methyl)silyl]oxy}methyl)-3-oxopropanoic acid"),
]


def _row(smiles, **flags):
    return Orthonym(style="pin", **flags).name_tiered(smiles)


@pytest.mark.parametrize("cause, smiles, pin_name", CASES, ids=[f"{c}-{i}" for i, (c, _, _) in enumerate(CASES)])
def test_the_default_tier_gives_the_name_and_label_of_the_pin_only_tier(cause, smiles, pin_name):
    pin_row = _row(smiles)
    assert (pin_row["name"], pin_row["tier"]) == (pin_name, "pin_verified")
    default_row = _row(smiles, **_emit_tier_flags("best-effort"))
    assert (default_row["name"], default_row["tier"]) == (pin_row["name"], pin_row["tier"])
