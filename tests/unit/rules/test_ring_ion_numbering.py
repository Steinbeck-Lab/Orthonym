"""A fused ring cation takes the numbering of its own fused name (``rules/ring_ion_numbering.py``).

The ring-locant supplier numbered a fused ring that its catalogue misses with the orienter of a single ring, so
'C[n+]1cccn2ncnc21' was composed as '6-methyl[1,2,4]triazolo[1,5-a]pyrimidin-6-ium' (C-6 is a carbon), the round trip
refused it and the PIN tier declined. "Numbering fused ring systems" (the Blue Book) numbers the fused
parent; "General rule for systematically naming cationic centers in parent hydrides" (the Blue Book):
"A cation derived formally by adding one or more hydrons to any position of a neutral parent hydride... is named by
replacing the final letter 'e' of the parent hydride name... by the suffix 'ium'", the '-ium' at the substituted
nitrogen's own locant.
"""
import pytest
from rdkit import Chem

from orthonym.rules.ring_ion_numbering import bare_fused_ring_locants
from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.unit


def test_a_monocycle_and_a_substituted_ring_keep_the_supplier():
    assert bare_fused_ring_locants(Chem.MolFromSmiles("c1ccncc1")) is None
    assert bare_fused_ring_locants(Chem.MolFromSmiles("Cc1ccc2ncnn2c1")) is None
    assert bare_fused_ring_locants(None) is None


@pytest.mark.opsin_gate
def test_the_bare_fused_ring_is_numbered_by_its_name():
    mol = Chem.MolFromSmiles("c1cnc2ncnn2c1")          # [1,2,4]triazolo[1,5-a]pyrimidine
    locants = bare_fused_ring_locants(mol)
    assert locants is not None and len(locants) == mol.GetNumAtoms()
    # the pyrimidine N next to the fusion carbon is N-4, the bridgehead N is N-8
    n_pyrimidine = [a.GetIdx() for a in mol.GetAtoms()
                    if a.GetSymbol() == "N" and not a.IsInRingSize(5)]
    assert [locants[i] for i in n_pyrimidine] == [4]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    # was '5-methyl-1,5,7,9-tetraazabicyclo[4.3.0]nona-1,3,6,8-tetraen-1-ium' (systematic) and no PIN
    ("C[n+]1cccn2ncnc21", "4-methyl[1,2,4]triazolo[1,5-a]pyrimidin-4-ium"),
    ("CC[n+]1cccn2ncnc21", "4-ethyl[1,2,4]triazolo[1,5-a]pyrimidin-4-ium"),
    # unchanged
    ("C[n+]1ccn2ncnc2c1", "7-methyl[1,2,4]triazolo[1,5-a]pyrazin-7-ium"),
    ("C[n+]1nnc2cccnn21", "3-methyl[1,2,3,4]tetrazolo[1,5-b]pyridazin-3-ium"),
    ("C[n+]1cccc2ccccc21", "1-methylquinolin-1-ium"),
])
def test_fused_ring_cations(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
def test_an_aromatic_oxo_ring_cation_keeps_no_pin():
    # cut off, the '=O' of C-6 leaves a mancude bare purine whose name states none of the hydro prefixes the oxo
    # needs, the Blue Book; '2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-isoindol-2-ium (PIN)',:41447):
    # '2-amino-7,9-dimethyl-6-oxo-1H-purin-7-ium' was labelled pin_verified; the PIN tier declines as before
    from tests.support.pin_tiers import tier_row
    assert tier_row("C[n+]1cn(C)c2nc(N)[nH]c(=O)c21", "pin")["tier"] == "abstain"
    # the nucleoside keeps its retained name below the PIN, the Blue Book; gold P14C-)
    row = tier_row("C[n+]1cn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c2nc(N)[nH]c(=O)c21", "valid")
    assert (row["name"], row["is_pin"]) == ("7-methylguanosine", False)


@pytest.mark.opsin_gate
def test_a_bridged_cation_with_letter_locants_is_named():
    # the bare ring's numbering mixes 4 and '4a': its substituent locants are sorted as ring locants
    # (ions._p74_ring_substituent_prefix raised TypeError and the salt lost its name)
    from tests.support.pin_tiers import tier_row
    row = tier_row("C[N@@+]1(CC[C@]23C4C(=C)CCC2(C1CC5=C3C(=C(C=C5)O)O4)O)CC6CC6.[Br-]", "valid")
    assert row["name"] == ("(1S,4R)-4-(cyclopropylmethyl)-10,17-dihydroxy-4-methyl-14-methylidene-12-oxa-4-azapentacyclo"
                           "[9.6.1.0^1,13.0^5,17.0^7,18]octadeca-7(18),8,10-trien-4-ium bromide"), row


@pytest.mark.opsin_gate
def test_the_numbering_of_one_bare_ring_is_computed_once_per_naming(monkeypatch):
    # a cation naming asked for the numbering of one bare ring up to 222 times (time-out audit 2026-11-12)
    from orthonym.assembly.memo import pop_scope, push_scope
    from orthonym.rules import ring_ion_numbering as rin
    computed = []
    real = rin._named_locants
    monkeypatch.setattr(rin, "_named_locants", lambda *a: computed.append(1) or real(*a))
    mol = Chem.MolFromSmiles("c1cnc2ncnn2c1")
    token = push_scope()
    try:
        first = bare_fused_ring_locants(mol)
        first[0] = "edited"                         # a caller's edit does not reach the stored numbering
        second = bare_fused_ring_locants(Chem.Mol(mol))
    finally:
        pop_scope(token)
    assert len(computed) == 1 and second[0] != "edited"
