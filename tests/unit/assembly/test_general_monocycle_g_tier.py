""" (g) in the numbering of the general engine's all-carbon monocycle (LO7c).

``### **** NUMBERING`` (the Blue Book): "When several structural features appear in
cyclic and acyclic compounds, low locants are assigned to them in the following decreasing order of
seniority" (:3221), and criterion "(g) lowest locants for the substituent cited first as a prefix in
the name" (:3307); '1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene)' (:3318),
'1-ethyl-4-methylcyclohexane (PIN) [for numbering, see (g)]' (``### ****
ALPHANUMERICAL ORDER``, ``****``,:3457). The prefix that is cited first is the first one in
alphanumerical order: ``****`` "The name of a prefix for a substituent is considered to begin
with the first letter of its complete name" (:3477).

``general_engine._orient_carbocycle`` ranked the orientations of the ring by the principal group, the
ring unsaturation and the substituent locants, then by the canonical ranks, so a tie went to the SMILES
spelling: '4-(benzenesulfonyl)-1-methylbenzene' ('benzenesulfonyl', b, is cited before 'methyl', m, and
takes the lower locant) for one spelling of a molecule and the right locants for another. The
orientation now ties to (g) of the prefixes the ring carries (``heterocycles._ring_prefix_names``, the
source of the tier in every other ring numbering), as the heteromonocycle numbering of the same engine
has since fu-amb (``test_engine_prefix_ordering``).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.general_engine import name_general_monocycle
from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate  # the names below are built with the validity gate on

SULFONYL = "S(=O)(=O)c8ccccc8"        # 'benzenesulfonyl' (b)
THIOUREA = "C(=S)NCc8ccccc8"          # '[(benzylamino)(sulfanylidene)methyl]' (b, after 'benzenesulfonyl')
ETHANESULFONYL = "S(=O)(=O)CC"        # 'ethanesulfonyl' (e): a ring of its own would make the parent ambiguous
METHANESULFONYL = "S(C)(=O)=O"        # 'methanesulfonyl' (m, 'a' before the 'y' of 'methyl')


def _name(smiles):
    row = tier_row(smiles, "best-effort")
    return row["name"], row["tier"]


def _engine_monocycle(smiles):
    """The producer called directly (tests/unit/assembly/test_general_monocycle_pg_anchor.py)."""
    nm = Orthonym(style="pin", general_fallback=True, allow_aromatic_general=True)
    mol = Chem.MolFromSmiles(smiles)
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    return getattr(name_general_monocycle(mol, feats, allow_aromatic_general=True), "name", None)


def _ring(template, a, b):
    return template.replace("(A)", f"({a})").replace("(B)", f"({b})")


@pytest.mark.parametrize("template,sulfonyl,other,expected", [
    # 'ethanesulfonyl' (e) before 'methyl' (m)
    ("c1(A)ccc(B)cc1", ETHANESULFONYL, "C", "1-(ethanesulfonyl)-4-methylbenzene"),
    ("c1(A)cccc(B)c1", ETHANESULFONYL, "C", "1-(ethanesulfonyl)-3-methylbenzene"),
    ("c1(A)c(B)cccc1", ETHANESULFONYL, "C", "1-(ethanesulfonyl)-2-methylbenzene"),
    ("C1(A)CCC(B)CC1", ETHANESULFONYL, "C", "1-(ethanesulfonyl)-4-methylcyclohexane"),
    # 'methanesulfonyl' (m-e-t-h-a) before 'methyl' (m-e-t-h-y)
    ("c1(A)ccc(B)cc1", METHANESULFONYL, "C", "1-(methanesulfonyl)-4-methylbenzene"),
])
@pytest.mark.parametrize("swap", [False, True])
def test_the_prefix_cited_first_takes_locant_1_whichever_atom_the_smiles_starts_at(
        template, sulfonyl, other, expected, swap):
    a, b = (sulfonyl, other) if not swap else (other, sulfonyl)
    smiles = _ring(template, a, b)
    assert _engine_monocycle(smiles) == expected
    name, tier = _name(smiles)
    assert name == expected
    assert tier == "systematic_verified"


@pytest.mark.parametrize("smiles,expected", [
    ("C1(S(=O)(=O)c8ccccc8)CCC(C(=S)NCc8ccccc8)CCC1",
     "1-(benzenesulfonyl)-4-[(benzylamino)(sulfanylidene)methyl]cycloheptane"),
    ("C1(C(=S)NCc8ccccc8)CCC(S(=O)(=O)c8ccccc8)CCC1",
     "1-(benzenesulfonyl)-4-[(benzylamino)(sulfanylidene)methyl]cycloheptane"),
    ("C1(S(=O)(=O)c8ccccc8)CC(C(=S)NCc8ccccc8)CCCC1",
     "1-(benzenesulfonyl)-3-[(benzylamino)(sulfanylidene)methyl]cycloheptane"),
    ("C1(C(=S)NCc8ccccc8)CCCC(S(=O)(=O)c8ccccc8)CCC1",
     "1-(benzenesulfonyl)-5-[(benzylamino)(sulfanylidene)methyl]cyclooctane"),
])
def test_the_prefix_cited_first_takes_locant_1_of_a_large_cycloalkane(smiles, expected):
    assert _name(smiles) == (expected, "systematic_verified")


@pytest.mark.parametrize("smiles,expected", [
    # decided before (g): the principal characteristic group (c)), then the ring double bonds
    ("O=C1C=CC(=O)C=C1", "cyclohexa-2,5-diene-1,4-dione"),
    ("O=C1CCC(=O)CC1", "cyclohexane-1,4-dione"),
])
def test_a_numbering_decided_by_an_earlier_criterion_is_unchanged(smiles, expected):
    assert _engine_monocycle(smiles) == expected
