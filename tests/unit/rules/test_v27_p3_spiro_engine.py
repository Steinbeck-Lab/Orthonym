"""v27 Phase 3 — analyze_spiro_universal + audit_spiro_descriptor (P-24.2).

Reference SMILES seeded via OPSIN (name->structure) from the BB PIN examples
quoted in 
"""
import pytest
from rdkit import Chem

from orthonym.rules.vonbaeyer_universal import (
    analyze_spiro_universal, audit_spiro_descriptor, SpiroSystem,
)

# name -> (SMILES from OPSIN, expected descriptor, expected total_atoms)
MONOSPIRO = {
    "spiro[4.5]decane": ("C1CCCC12CCCCC2", "spiro[4.5]", 10),
    "spiro[4.4]nonane": ("C1CCCC12CCCC2", "spiro[4.4]", 9),
    "spiro[5.5]undecane": ("C1CCCCC12CCCCC2", "spiro[5.5]", 11),
}

POLYSPIRO = {
    # linear dispiro (Orthonym emits the audit-exact plain form which
    # round-trips; superscript variant is name-quality only)
    "dispiro[3.2.3.2]dodecane": ("C1CCC12CCC1(CCC1)CC2", 12),
    "dispiro[2.2.3.2]undecane": ("C1CC12CCC1(CCC1)CC2", 11),
}

HETEROSPIRO = {
    "6-oxaspiro[4.5]decane": ("C1CCCC12OCCCC2", "6-oxa", "spiro[4.5]", 10),
    "9-oxa-6-azaspiro[4.5]decane": ("C1CCCC12NCCOC2", "9-oxa-6-aza", "spiro[4.5]", 10),
    "7-thia-9-azaspiro[4.5]decane": ("C1CCCC12CSCNC2", "7-thia-9-aza", "spiro[4.5]", 10),
}


@pytest.mark.unit
@pytest.mark.parametrize("name,data", MONOSPIRO.items())
def test_monospiro(name, data):
    smi, desc, total = data
    mol = Chem.MolFromSmiles(smi)
    sp = analyze_spiro_universal(mol)
    assert sp is not None, f"{name}: analyzer refused"
    assert sp.descriptor == desc
    assert sp.total_atoms == total
    assert sp.hetero_prefix == ""
    # numbering covers exactly the ring atoms
    assert set(sp.atom_to_locant.keys()) == set(sp.cage_atoms)
    assert sorted(sp.atom_to_locant.values()) == list(range(1, total + 1))


@pytest.mark.unit
@pytest.mark.parametrize("name,data", POLYSPIRO.items())
def test_polyspiro(name, data):
    smi, total = data
    mol = Chem.MolFromSmiles(smi)
    sp = analyze_spiro_universal(mol)
    assert sp is not None, f"{name}: analyzer refused"
    assert sp.total_atoms == total
    assert sp.descriptor.startswith("dispiro[")
    assert sorted(sp.atom_to_locant.values()) == list(range(1, total + 1))


@pytest.mark.unit
@pytest.mark.parametrize("name,data", HETEROSPIRO.items())
def test_heterospiro(name, data):
    smi, het, desc, total = data
    mol = Chem.MolFromSmiles(smi)
    sp = analyze_spiro_universal(mol)
    assert sp is not None, f"{name}: analyzer refused"
    assert sp.descriptor == desc
    assert sp.hetero_prefix == het, f"{name}: got hetero_prefix {sp.hetero_prefix!r}"
    assert sp.total_atoms == total


@pytest.mark.unit
def test_refuses_von_baeyer_bridged():
    # norbornane (bicyclo[2.2.1]heptane) is NOT spiro -> None (routing invariant)
    mol = Chem.MolFromSmiles("C1CC2CCC1C2")
    assert analyze_spiro_universal(mol) is None


@pytest.mark.unit
def test_refuses_fused():
    # decalin (fused) is not spiro -> None
    mol = Chem.MolFromSmiles("C1CCC2CCCCC2C1")
    assert analyze_spiro_universal(mol) is None


@pytest.mark.unit
def test_refuses_aromatic_spiro_without_mancude():
    # spiro[cyclohexane-1,1'-indene] carries an aromatic ring -> fail closed
    mol = Chem.MolFromSmiles("C12(C=CC3=CC=CC=C13)CCCCC2")
    assert analyze_spiro_universal(mol, allow_mancude=False) is None


# ---- audit_spiro_descriptor direct tests ----

@pytest.mark.unit
def test_audit_accepts_correct():
    mol = Chem.MolFromSmiles("C1CCCC12CCCCC2")  # spiro[4.5]decane
    sp = analyze_spiro_universal(mol)
    assert sp is not None
    # reconstruct spiro atoms in the (sorted-order) submol space is internal;
    # re-run the audit against the returned analysis mapped back to mol space
    numbering = sp.atom_to_locant
    ri = mol.GetRingInfo()
    spiro = {a for a in range(mol.GetNumAtoms())
             if ri.NumAtomRings(a) == 2}
    assert audit_spiro_descriptor(
        mol, set(sp.cage_atoms), numbering, spiro, sp.descriptor) is True


@pytest.mark.unit
def test_audit_rejects_duplicate_locant():
    mol = Chem.MolFromSmiles("C1CCCC12CCCCC2")
    ri = mol.GetRingInfo()
    spiro = {a for a in range(mol.GetNumAtoms()) if ri.NumAtomRings(a) == 2}
    cage = set(range(mol.GetNumAtoms()))
    # a numbering that collapses two atoms to the same locant
    bad = {i: (i + 1) for i in range(mol.GetNumAtoms())}
    bad[1] = bad[0]  # duplicate
    assert audit_spiro_descriptor(mol, cage, bad, spiro, "spiro[4.5]") is False


@pytest.mark.unit
def test_audit_rejects_wrong_size_descriptor():
    mol = Chem.MolFromSmiles("C1CCCC12CCCCC2")  # really spiro[4.5], 10 atoms
    ri = mol.GetRingInfo()
    spiro = {a for a in range(mol.GetNumAtoms()) if ri.NumAtomRings(a) == 2}
    cage = set(range(mol.GetNumAtoms()))
    good = {i: i + 1 for i in range(mol.GetNumAtoms())}  # bijection
    # descriptor arithmetic 3+4+1 = 8 != 10 -> reject
    assert audit_spiro_descriptor(mol, cage, good, spiro, "spiro[3.4]") is False


# ---- Task 5: parent producer (name_general_spiro) via full perceive+classify ----

@pytest.fixture(scope="module")
def _namer():
    from orthonym.namer import Orthonym
    return Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)


def _spiro_parent(nm, smi):
    from orthonym.assembly.general_engine import name_general_spiro
    mol = Chem.MolFromSmiles(smi)
    f = nm._perceive(mol, smi, Chem.MolToSmiles(mol))
    nm._classify(f)
    r = name_general_spiro(mol, f, allow_aromatic_general=True,
                           allow_charged=True)
    return r.name if r else None


@pytest.mark.unit
@pytest.mark.parametrize("smi,expected", [
    ("OC(=O)C1CCCC12CCCCC2", "spiro[4.5]decane-1-carboxylic acid"),
    # positions 2 and 3 are symmetry-equivalent; PIN cites the lowest locant
    ("N#CC1CCC2(CCCCC2)C1", "spiro[4.5]decane-2-carbonitrile"),
    ("C1(CCCC12CCCCC2)O", "spiro[4.5]decan-1-ol"),
    ("C1(CCCC12CCCCC2)=O", "spiro[4.5]decan-1-one"),
    ("C1CCCC12OCCCC2", "6-oxaspiro[4.5]decane"),
])
def test_spiro_parent_producer(_namer, smi, expected):
    assert _spiro_parent(_namer, smi) == expected


@pytest.mark.unit
def test_spiro_parent_refuses_non_spiro(_namer):
    # decalin (fused) must NOT be produced by the spiro path
    assert _spiro_parent(_namer, "C1CCC2CCCCC2C1") is None


# ---- Task 6: spiro-as-yl substituent (hetero / polyspiro), complete tier ----

def _ring_frag_attach(mol):
    """Extract the ring system + the ring atom bearing an exocyclic C (the
    free-valence attachment) for a single-ring-system substituent probe."""
    from orthonym.rules.ring_substituents import _extract_ring_submol
    ring_atoms = tuple(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    attach = None
    for a in ring_atoms:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if not nb.IsInRing() and nb.GetSymbol() == 'C':
                attach = a
    return _extract_ring_submol(mol, ring_atoms, attach)


@pytest.mark.unit
@pytest.mark.parametrize("smi,expected", [
    # hetero spiro substituent (was carbocyclic-only -> "substituent")
    ("C1OCCC12CCC(CC2)CC(=O)O", "2-oxaspiro[4.5]decan-8-yl"),
    # polyspiro substituent. v39 Task F round 2: this dispiro skeleton's two
    # equal-length middle-ring arcs (both 2 carbons) are a genuine P-24.2.2
    # numbering tie the descriptor/spiro-atom-locant rules do not resolve
    # (same descriptor "dispiro[3.2.3.2]", same spiro-atom locants {4,7}
    # either way) -- P-31.1.4.3.4 (lowest locant to the free valence) then
    # picks locant 5 over the old code's arbitrary 12. Both denote the
    # IDENTICAL molecule (confirmed: OPSIN-parsing "dispiro[3.2.3.2]dodecan-
    # 5-yl"acetic acid and the -12-yl form give the same InChIKey,
    # HCHGJBAWDHMYAZ-UHFFFAOYSA-N -- a real molecular symmetry, not a bug),
    # so this is a PIN correction, not a behavior regression.
    ("C1CCC12CCC1(CCC1)CC2CC(=O)O", "dispiro[3.2.3.2]dodecan-5-yl"),
])
def test_universal_spiro_substituent(smi, expected):
    """Java-free unit test of the P3 spiro `-yl` producer (the fragment namer),
    independent of the whole-molecule SELF-01 gate."""
    from orthonym.rules.ring_substituents import (
        _universal_spiro_substituent_name,
    )
    mol = Chem.MolFromSmiles(smi)
    sub, attach_sub = _ring_frag_attach(mol)
    assert sub is not None
    got = _universal_spiro_substituent_name(sub, attach_sub, allow_mancude=True)
    assert got == expected


@pytest.mark.unit
def test_spiro_substituent_gated_off_default():
    """PIN default is byte-identical: the P3 spiro `-yl` producer is inert
    (allow_mancude default False -> carbocyclic-only path, which declines a
    HETERO spiro fragment)."""
    from orthonym.rules.ring_substituents import (
        _universal_spiro_substituent_name,
    )
    mol = Chem.MolFromSmiles("C1OCCC12CCC(CC2)CC(=O)O")
    sub, attach_sub = _ring_frag_attach(mol)
    # the universal namer itself refuses aromatic-only-flagged mancude; here the
    # gating is at the _polycyclic_substituent_name call site (allow_mancude),
    # so the direct namer with allow_mancude=False still names the (non-aromatic)
    # hetero spiro -> assert the CALL SITE gating instead.
    from orthonym.rules.ring_substituents import _polycyclic_substituent_name
    ring_atoms = tuple(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    attach = next(a for a in ring_atoms
                  for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                  if not nb.IsInRing() and nb.GetSymbol() == 'C')
    off = _polycyclic_substituent_name(mol, ring_atoms, attach,
                                       allow_mancude=False)
    on = _polycyclic_substituent_name(mol, ring_atoms, attach,
                                      allow_mancude=True)
    assert off is None            # PIN default: declined (byte-identical)
    assert on == "2-oxaspiro[4.5]decan-8-yl"
