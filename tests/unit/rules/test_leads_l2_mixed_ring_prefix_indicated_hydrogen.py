"""Leads L2 (E), item N8a (fix round): a compound substituent prefix is spelled in full, indicated
hydrogen and hyphen included.

Indicated hydrogen: ``**** Indicated hydrogen`` (the Blue Book) and the paragraph after its
pyrrole examples (:3721): "In general nomenclature, indicated hydrogen may be omitted... However, in a
preferred IUPAC name a locant and the symbol 'H' must be cited". A substituent group keeps it:
'(1H-indol-1-yl)acetic acid (PIN)' table,:2039). So the N-attached azole that carries a ring
substituent is '4-phenyl-1H-1,2,3-triazol-1-yl', '5-phenyl-1H-tetrazol-1-yl', '4-phenyl-1H-imidazol-1-yl'.

Hyphen: ``**** Hyphens are used in substitutive names: (a) to separate locants from words or
word fragments`` (the Blue Book,:6938). A stem that begins with a locant is set off from the prefix:
'4-phenyl-1,3,5-triazin-2-yl', never '4-phenyl1,3,5-triazin-2-yl'.

``name_mixed_ring_prefix`` wrote its parent as a bare stem, with no hyphen: '4-phenyl1,2,3-triazol-1-yl'
once the one numbering fix made its locants right, '4-phenylimidazol-1-yl', '4-phenyl1,3,5-triazin-2-yl'.
It also reads a pyrazole as 'imidazole'. Before the numbering fix its wrong locants were rejected by the
round-trip check and a ring-by-ring producer, which cites the indicated hydrogen and the hyphen, named
the ring; with right locants the malformed names survived the round-trip and replaced the well-formed
ones. The producer now declines a mancude parent that has a pyrrole-type atom, so the ring-by-ring
producers name it, and it sets the hyphen before a stem that begins with a locant.
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.perception.rings import get_ring_systems
from orthonym.rules import ring_assemblies as ra
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

#: (id, SMILES, the PIN)
ROWS = [
    ("triazole-123-4-phenyl", "CC(O)Cn1nnc(c1)-c1ccccc1",
     "1-(4-phenyl-1H-1,2,3-triazol-1-yl)propan-2-ol"),
    ("triazole-123-4-phenyl-amine", "NCCn1nnc(c1)-c1ccccc1",
     "2-(4-phenyl-1H-1,2,3-triazol-1-yl)ethan-1-amine"),
    ("triazole-123-5-phenyl", "OCCn1nncc1-c1ccccc1",
     "2-(5-phenyl-1H-1,2,3-triazol-1-yl)ethan-1-ol"),
    ("triazole-124-5-phenyl", "OCCn1ncnc1-c1ccccc1",
     "2-(5-phenyl-1H-1,2,4-triazol-1-yl)ethan-1-ol"),
    ("triazole-124-3-phenyl", "OCCn1nc(nc1)-c1ccccc1",
     "2-(3-phenyl-1H-1,2,4-triazol-1-yl)ethan-1-ol"),
    ("triazole-124-4-yl", "OCCn1cnnc1-c1ccccc1",
     "2-(3-phenyl-4H-1,2,4-triazol-4-yl)ethan-1-ol"),
    ("tetrazole-1-yl", "OCCn1nnnc1-c1ccccc1", "2-(5-phenyl-1H-tetrazol-1-yl)ethan-1-ol"),
    ("tetrazole-2-yl", "OCCn1nc(nn1)-c1ccccc1", "2-(5-phenyl-2H-tetrazol-2-yl)ethan-1-ol"),
    ("imidazole-4-phenyl", "OCCn1cnc(c1)-c1ccccc1", "2-(4-phenyl-1H-imidazol-1-yl)ethan-1-ol"),
    ("imidazole-2-phenyl", "OCCn1ccnc1-c1ccccc1", "2-(2-phenyl-1H-imidazol-1-yl)ethan-1-ol"),
    ("imidazole-2-pyridyl", "OCCn1ccnc1-c1ccncc1", "2-[2-(pyridin-4-yl)-1H-imidazol-1-yl]ethan-1-ol"),
    ("pyrrole-2-phenyl", "OCCn1cccc1-c1ccccc1", "2-(2-phenyl-1H-pyrrol-1-yl)ethan-1-ol"),
    ("pyrrole-3-phenyl", "OCCn1ccc(c1)-c1ccccc1", "2-(3-phenyl-1H-pyrrol-1-yl)ethan-1-ol"),
    ("pyrrole-2-pyridyl", "OCCn1cccc1-c1ccncc1", "2-[2-(pyridin-4-yl)-1H-pyrrol-1-yl]ethan-1-ol"),
    # the stem 'imidazole' also named a pyrazole in this producer
    ("pyrazole-5-phenyl", "OCCn1nccc1-c1ccccc1", "2-(5-phenyl-1H-pyrazol-1-yl)ethan-1-ol"),
    ("pyrazole-3-phenyl", "OCCn1nc(cc1)-c1ccccc1", "2-(3-phenyl-1H-pyrazol-1-yl)ethan-1-ol"),
    # the same parent attached through a carbon
    ("imidazol-2-yl", "OCCc1nc(-c2ccccc2)c[nH]1", "2-(4-phenyl-1H-imidazol-2-yl)ethan-1-ol"),
    ("pyrazol-3-yl", "OCCc1cc(-c2ccccc2)[nH]n1", "2-(5-phenyl-1H-pyrazol-3-yl)ethan-1-ol"),
    ("pyrrol-2-yl", "OCCc1ccc(-c2ccccc2)[nH]1", "2-(5-phenyl-1H-pyrrol-2-yl)ethan-1-ol"),
    # as the substituent of an acid (the prefix of a retained acid name)
    ("triazole-acetic-acid", "OC(=O)Cn1nnc(c1)-c1ccccc1", "(4-phenyl-1H-1,2,3-triazol-1-yl)acetic acid"),
    ("imidazole-acetic-acid", "OC(=O)Cn1cnc(c1)-c1ccccc1", "(4-phenyl-1H-imidazol-1-yl)acetic acid"),
    ("tetrazole-acetic-acid", "OC(=O)Cn1nnnc1-c1ccccc1", "(5-phenyl-1H-tetrazol-1-yl)acetic acid"),
    # no pyrrole-type atom, a stem that begins with a locant: the prefix is built here, with its hyphen
    ("triazine-135-2-yl", "OCCc1ncnc(n1)-c1ccccc1", "2-(4-phenyl-1,3,5-triazin-2-yl)ethan-1-ol"),
    ("triazine-124-3-yl", "OCCc1nnc(cn1)-c1ccccc1", "2-(6-phenyl-1,2,4-triazin-3-yl)ethan-1-ol"),
    ("triazine-124-5-yl", "OCCc1cnnc(n1)-c1ccccc1", "2-(3-phenyl-1,2,4-triazin-5-yl)ethan-1-ol"),
    ("triazine-124-6-yl", "OCCc1nncnc1-c1ccccc1", "2-(5-phenyl-1,2,4-triazin-6-yl)ethan-1-ol"),
]
IDS = [r[0] for r in ROWS]

#: parents without a pyrrole-type atom keep this producer: the prefix is still built here
CONTROLS = [
    ("pyridine", "OCCc1cc(-c2ccccc2)ccn1", "2-(4-phenylpyridin-2-yl)ethan-1-ol"),
    ("thiophene", "OCCc1ccc(-c2ccccc2)s1", "2-(5-phenylthiophen-2-yl)ethan-1-ol"),
    ("furan", "OCCc1ccc(-c2ccccc2)o1", "2-(5-phenylfuran-2-yl)ethan-1-ol"),
    ("piperidine", "OCCN1CCC(CC1)c1ccccc1", "2-(4-phenylpiperidin-1-yl)ethan-1-ol"),
]


def _prefix(mapped_smiles):
    """``name_mixed_ring_prefix`` for the ring system that holds the atom mapped 1 (the free valence)."""
    mol = Chem.MolFromSmiles(mapped_smiles)
    attach = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1)
    systems = get_ring_systems(mol, include_spiro=False)
    return ra.name_mixed_ring_prefix(
        mol, systems, ra._find_inter_system_bonds(mol, systems), attach)


# ------------------------------------------------------------------ the names read back
@pytest.mark.parametrize("_id,smiles,pin", ROWS + CONTROLS, ids=IDS + [c[0] for c in CONTROLS])
def test_the_pin_reads_back_to_the_full_inchikey(_id, smiles, pin):
    assert_full_rt(pin, smiles)


# ------------------------------------------------------------------ the producer alone
@pytest.mark.parametrize("mapped", [
    "OCC[n:1]1cnc(c1)-c1ccccc1",         # imidazol-1-yl
    "OCC[n:1]1nnc(c1)-c1ccccc1",         # 1,2,3-triazol-1-yl
    "OCC[n:1]1nnnc1-c1ccccc1",           # tetrazol-1-yl
    "OCC[n:1]1cccc1-c1ccccc1",           # pyrrol-1-yl
    "OCC[n:1]1nccc1-c1ccccc1",           # pyrazol-1-yl (read as 'imidazole' by the ring identifier)
    "OCC[c:1]1nc(-c2ccccc2)c[nH]1",      # a carbon-attached azole: the ring still needs its 1H
    "OCC[c:1]1ccc(-c2ccccc2)[nH]1",
])
def test_a_mancude_parent_with_a_pyrrole_type_atom_is_left_to_the_ring_by_ring_producers(mapped):
    assert _prefix(mapped) is None


@pytest.mark.parametrize("mapped,prefix", [
    ("OCC[c:1]1cc(-c2ccccc2)ccn1", "(4-phenylpyridin-2-yl)"),
    ("OCC[c:1]1ccc(-c2ccccc2)s1", "(5-phenylthiophen-2-yl)"),
    ("OCC[c:1]1ccc(-c2ccccc2)o1", "(5-phenylfuran-2-yl)"),
    ("OCC[N:1]1CCC(CC1)c1ccccc1", "(4-phenylpiperidin-1-yl)"),
    ("OCC[N:1]1CCN(CC1)c1ccccc1", "(4-phenylpiperazin-1-yl)"),
    # a stem that begins with a locant is set off from the prefix by a hyphen (a))
    ("OCC[c:1]1ncnc(n1)-c1ccccc1", "(4-phenyl-1,3,5-triazin-2-yl)"),
    ("OCC[c:1]1nnc(cn1)-c1ccccc1", "(6-phenyl-1,2,4-triazin-3-yl)"),
])
def test_a_parent_without_a_pyrrole_type_atom_is_still_named_here(mapped, prefix):
    assert _prefix(mapped) == prefix


# ------------------------------------------------------------------ the indicated-hydrogen reader
@pytest.mark.parametrize("smiles,default,substituted", [
    ("c1cc[nH]c1", [3], [3]),             # 1H-pyrrole
    ("Cn1cccc1", [], [1]),                  # an N-substituted pyrrole: the atom keeps no H
    ("c1ccncc1", [], []),                 # pyridine: the nitrogen is in a ring double bond
    ("c1ccsc1", [], []),                  # divalent S never bears an indicated hydrogen
    ("C1=CC=COC1", [5], [5]),             # 2H-pyran: the CH2
])
def test_include_substituted_keeps_the_atom_a_substituent_took_the_hydrogen_from(smiles, default, substituted):
    mol = Chem.MolFromSmiles(smiles)
    ring = list(mol.GetRingInfo().AtomRings()[0])
    assert sorted(ra._ring_indicated_h_atoms(mol, ring)) == default
    assert sorted(ra._ring_indicated_h_atoms(mol, ring, include_substituted=True)) == substituted


# ------------------------------------------------------------------ the engine
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namers():
    with jvm_slots(1, purpose="leads-L2-test"):
        yield {"pin": Orthonym(), "best-effort": Orthonym(style="pin", **_emit_tier_flags("best-effort"))}


@pytest.mark.parametrize("_id,smiles,pin", ROWS + CONTROLS, ids=IDS + [c[0] for c in CONTROLS])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_engine_cites_the_indicated_hydrogen(namers, _id, smiles, pin, tier):
    row = namers[tier].name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert not row.get("spelling_failures"), row


_ORDER_ROWS = [r for r in ROWS if r[0] in (
    "triazole-123-4-phenyl-amine", "triazole-123-5-phenyl", "triazole-123-4-phenyl", "tetrazole-1-yl",
    "imidazole-4-phenyl", "pyrrole-2-phenyl", "triazine-135-2-yl", "triazine-124-6-yl")]


@pytest.mark.parametrize("_id,smiles,pin", _ORDER_ROWS, ids=[r[0] for r in _ORDER_ROWS])
def test_the_name_does_not_depend_on_the_atom_order(namers, _id, smiles, pin):
    mol = Chem.MolFromSmiles(smiles)
    orders = {smiles} | {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(10)}
    names = {namers["pin"].name_tiered(o)["name"] for o in orders}
    assert names == {pin}
