"""Three-ring fusion names: the parent component follows in full.

 fix a performance pass (wp4-ring-suffix), research items F11a/F11b/F11c. The bounded
generate-and-test namer (``fused_rings._name_ortho_fused_generate_and_test``)
ranked parent components by (a) and (b) only, could take a fusion-named
pair ('thieno[3,2-b]thiophene') as the parent component, and never offered the
indicated-hydrogen or '[1,3]oxazolo' candidates the PIN needs. Every expected name
below is an OPSIN 2.9.0 round trip to the input's full standard InChIKey, and for the
indicated-hydrogen names also to the input's canonical SMILES (the standard InChI does
not fix which ring atom carries the mobile hydrogen); both checks are made here outside
the engine (``tests/support/rt_assert.py`` fresh parse).

Rules (the Blue Book), "Seniority criteria for selecting the parent
component" (:12133): "(a) a component containing at least one of the heteroatoms
occurring earlier in the following order: N > F >..." (:12139); "(b) a component
containing the greater number of rings" (:12163); "(c) A component containing the
larger ring at the first point of difference when comparing rings in order of
decreasing size" (:12234, '2H-furo[3,2-b]pyran (PIN) [pyran (6 ring) preferred to
furan (5 ring)]':12246); "(d) A component containing the greater number of
heteroatoms of any kind" (:12260); "(h) A component with the lower locants for
heteroatoms" (:12336, 'pyrazino[2,3-d]pyridazine (PIN)'). (:11907): the
heteroatom locants of a component are "enclosed within square brackets".
(:11885): the parent component is one ring or ring system ("Its name is never
modified"), so a fused pair named with a fusion descriptor is not a component; the
PIN of such a system needs primed higher-order locants, which this
namer does not build. row (2) (:11521-11523): the phenanthrolines
are retained names ("the PIN is 1,7-phenanthroline; other isomers are: 1,8-; 1,9-;
1,10-; 2,7-; 2,8-; 2,9-; 3,7-; 3,8-; 4,7-").
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from tests.support.rt_assert import _independent_parse, name_best_effort, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


PIN_ROWS = [
    # F11a (d): quinoxaline (2 heteroatoms) over quinoline (1); was
    # 'pyrazino[2,3-g]quinoline' (the tie fell to the string order)
    ("c1cnc2cc3nccnc3cc2c1", "pyrido[2,3-g]quinoxaline"),
    # F11a (c): quinoline (6,6) over indole (6,5); was '3H-pyrido[3,2-e]indole'
    ("c1cnc2ccc3[nH]ccc3c2c1", "3H-pyrrolo[3,2-f]quinoline"),
    # F11a (c); was '1H-pyrido[3,2-g]indole'
    ("c1cnc2c(c1)ccc1cc[nH]c12", "1H-pyrrolo[3,2-h]quinoline"),
    # F11c (i): the indicated-hydrogen candidates were never offered when the parser
    # completed the hydrogen-free name with a CH2; were '1H-pyrido[3,2-f]indole',
    # '1H-quinolino[2,3-b]pyrrole' (violates (b)), '1H-pyrido[2,3-f]benzimidazole',
    # '1H-pyrido[4,3-f]indole', '1H-pyrido[2,3-f]indole'
    ("c1cnc2cc3[nH]ccc3cc2c1", "1H-pyrrolo[3,2-g]quinoline"),
    ("c1ccc2nc3[nH]ccc3cc2c1", "1H-pyrrolo[2,3-b]quinoline"),
    ("c1cnc2cc3nc[nH]c3cc2c1", "1H-imidazo[4,5-g]quinoline"),
    ("c1cc2cc3cc[nH]c3cc2cn1", "1H-pyrrolo[3,2-g]isoquinoline"),
    ("c1cnc2cc3cc[nH]c3cc2c1", "1H-pyrrolo[2,3-g]quinoline"),
    # F11c (ii): '[1,3]oxazolo' / '[1,3]thiazolo' as the attached component;
    # were 'pyrido[3,2-f][1,3]benzoxazole' / 'pyrido[3,2-f][1,3]benzothiazole'
    ("c1cnc2cc3ocnc3cc2c1", "[1,3]oxazolo[4,5-g]quinoline"),
    ("c1cnc2cc3scnc3cc2c1", "[1,3]thiazolo[4,5-g]quinoline"),
    # (h): quinoline (N at 1) over isoquinoline (N at 2); was 'pyrido[2,3-g]isoquinoline'
    ("c1cnc2cc3ccncc3cc2c1", "pyrido[3,4-g]quinoline"),
    # (:11911) "To the letter... are prefixed, if necessary, the numbers
    # of the positions of attachment": never for benzo ('benzo[g]isoquinoline',
    #:11909); were 'benzo[1,2-f]isoquinoline', '1H-benzo[1,2-g]indole',...
    ("c1ccc2c(c1)ccc1cnccc12", "benzo[f]isoquinoline"),
    ("c1ccc2c(c1)ccc1cc[nH]c12", "1H-benzo[g]indole"),
    ("c1ccc2c(c1)ccc1ccnnc12", "benzo[h]cinnoline"),
    # (:13435,:13437) a benzoheterocycle is a component only "in which the
    # benzene ring is not part of a system having a retained name such as quinoline
    # or naphthalene"; "Retained names are senior to names of
    # benzoheterocycles" (:13455). Were '3H-benzo[1,2-e]benzimidazole', 'benzo[1,2-g]
    # [1]benzofuran', 'benzo[1,2-e][1,3]benzoxazole'
    ("c1ccc2c(c1)ccc1[nH]cnc12", "3H-naphtho[1,2-d]imidazole"),
    ("c1ccc2c(c1)ccc1ccoc12", "naphtho[1,2-b]furan"),
    ("c1ccc2c(c1)ccc1ocnc12", "naphtho[1,2-d][1,3]oxazole"),
    # (:13439) a benzoheterocycle as the parent, 'thieno[3,2-f][2,1]
    # benzothiazole (PIN)': O > S, so [1]benzofuran over [1]benzothiophene
    ("c1cc2cc3ccsc3cc2o1", "thieno[3,2-f][1]benzofuran"),
    # unchanged (commit 2ae3b9820's rows keep their parents under the full key)
    ("c1ccc2c(c1)oc1ncccc12", "[1]benzofuro[2,3-b]pyridine"),
    ("c1ccc2c(c1)[nH]c1ncccc12", "9H-pyrido[2,3-b]indole"),
]


def _same_tautomer(name: str, smiles: str) -> bool:
    parsed = _independent_parse(name)
    m = Chem.MolFromSmiles(parsed) if parsed else None
    return m is not None and Chem.MolToSmiles(m) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles,pin", PIN_ROWS, ids=[r[0] for r in PIN_ROWS])
def test_fusion_parent_component(smiles, pin):
    name = name_compound(smiles)
    assert name == pin
    assert name_is_rt_exact(name, smiles), f"{name!r} does not round-trip to {smiles}"
    assert _same_tautomer(name, smiles), f"{name!r} puts the hydrogen elsewhere"


# F11b: the only parent this namer could offer was a fusion-named pair; the PIN
# ("thieno[2',3':4,5]thieno[2,3-b]pyridine", "furo[2',3':4,5]furo[2,3-b]pyridine")
# needs primed higher-order locants. The phenanthroline isomers not in the catalog
# have a retained PIN. None of these may be labelled pin_verified; the best-effort
# tier still names each one round-trip exact.
NOT_PIN_ROWS = [
    ("c1cnc2sc3ccsc3c2c1", "pyrido[3,2-d]thieno[3,2-b]thiophene"),
    ("c1cnc2c(c1)sc1sccc12", "pyrido[2,3-d]thieno[2,3-b]thiophene"),
    ("c1cnc2oc3ccoc3c2c1", "pyrido[3,2-d]furo[3,2-b]furan"),
    ("c1cnc2c(c1)ccc1ccncc12", "pyrido[3,2-h]isoquinoline"),
    ("c1cnc2ccc3cnccc3c2c1", "pyrido[3,2-f]isoquinoline"),
    # (:13451) "A multiparent name is preferred to a fused ring system,
    # when there is a choice": 'benzo[1,2-b:4,5-c']difuran (PIN) (not furo[3,4-f]
    # [1]benzofuran'; the multiparent name is not built here
    ("c1cc2cc3ccoc3cc2o1", "furo[3,2-f][1]benzofuran"),
]


@pytest.mark.parametrize("smiles,old", NOT_PIN_ROWS, ids=[r[0] for r in NOT_PIN_ROWS])
def test_unbuildable_parent_is_not_pin_verified(smiles, old):
    res = Orthonym(style="pin").name_tiered(smiles)
    assert res.get("tier") != "pin_verified", res
    assert res.get("name") != old, res
    be = name_best_effort(smiles)
    assert be.get("name") and name_is_rt_exact(be["name"], smiles), be
