""" sub-lever A: S-rooted thioether substituent with a RING-bearing R.

The chalcogen-rooted sulfanyl cascade tier declines a saturated-ring / ring-on-chain
R, so -S-cyclohexyl / -S-CH2Ar fell through to the ugly replacement name. Under the
best-effort tier recurse name_substituent on R and wrap 'sulfanyl' -- mirroring the
O-rooted alkoxy (cyclohexyloxy) and N-rooted amino (cyclohexylamino) ring paths.
Gate-independent probe re-anchor rejects a yl-less ring-assembly R (the ed52fa98 /
8afa533c F1 class).

 a phase Task 10 (E2a, ring-branch ownership) GRADUATED this from
best-effort-only to the PIN default too: `name_substituent_fragment`'s Step 1b
(chalcogen-ether substituent, `assembly/substituent_naming.py`) was widened from
``('Se', 'Te')`` to include plain ``'S'``, so a direct ``-S-Ar``/``-S-cycloalkyl``
attachment is now named through the SAME chalcogen-agnostic `get_sulfanyl_prefix`
builder the best-effort tier already used -- which recurses the ring side through
the SAME trustworthy ring chokepoint (`rules.ring_substituents
.name_ring_system_substituent`) `name_substituent_fragment`'s own PIN-tier ring
branches already rely on elsewhere, not the mancude-gated best-effort path. Verified
byte-identical to `_name_thio_ring_branch`'s answer on every case here, INCLUDING the
adversarial ring-assembly probe-reanchor canary
(`test_ring_assembly_R_names_when_it_round_trips`, unaffected -- it calls
`_name_thio_ring_branch` directly). The former "PIN keeps the historical sentinel"
scope limit was a not-yet-built gap, not a correctness objection; `allow_mancude`
no longer gates this specific shape (both states now reach the identical, RT-verified
name) because `name_substituent_fragment` itself has no gate parameter to honour --
it is PIN-tier by construction, called identically regardless of the caller's mancude
setting.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent


def _sulfanyl_frag(mol_smi):
    """Build (mol, frag_atoms, attach_S) for a -S-R substituent from a WHOLE
    molecule whose S bridges a REAL parent atom and R. A `[*]` dummy parent breaks
    the helper's CH3-S-R probe (RDKit valence artifact at the wildcard bond), so
    the substituent path must be exercised with a real parent atom."""
    m = Chem.MolFromSmiles(mol_smi)
    assert m is not None, mol_smi
    # the divalent bridging S: neutral, degree 2, both single bonds
    S = next(a.GetIdx() for a in m.GetAtoms()
             if a.GetSymbol() == 'S' and a.GetDegree() == 2
             and a.GetFormalCharge() == 0)
    # R side = the S neighbour that is NOT the aromatic-parent ring anchor; grab
    # the whole substituent fragment by BFS from S excluding the parent side.
    nbrs = [n.GetIdx() for n in m.GetAtomWithIdx(S).GetNeighbors()]
    # parent side = the neighbour in the largest aromatic ring (the tolyl anchor)
    ri = m.GetRingInfo()
    parent = next(n for n in nbrs if ri.NumAtomRings(n) > 0
                  and m.GetAtomWithIdx(n).GetIsAromatic())
    rside = next(n for n in nbrs if n != parent)
    frag = {S}
    stack = [rside]
    while stack:
        a = stack.pop()
        if a in frag:
            continue
        frag.add(a)
        for nb in m.GetAtomWithIdx(a).GetNeighbors():
            ni = nb.GetIdx()
            if ni != S and ni not in frag and ni != parent:
                stack.append(ni)
    return m, sorted(frag), S


# whole-molecule SMILES (S bridges a tolyl parent and R), expected -S-R prefix.
BEST_EFFORT = [
    ("Cc1ccc(SC2CCCCC2)cc1", "cyclohexylsulfanyl"),
    ("Cc1ccc(SCc2ccccc2)cc1", "benzylsulfanyl"),
    ("Cc1ccc(Sc2ccccc2)cc1", "phenylsulfanyl"),
    ("Cc1ccc(SC2CCCC2)cc1", "cyclopentylsulfanyl"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("mol_smi,expected", BEST_EFFORT)
def test_best_effort_names_s_rooted_ring_sulfanyl(mol_smi, expected):
    m, frag, S = _sulfanyl_frag(mol_smi)
    assert name_substituent(m, frag, S, allow_mancude=True) == expected


@pytest.mark.parametrize("mol_smi,expected", BEST_EFFORT)
def test_pin_default_byte_identical(mol_smi, expected):
    # a phase Task 10: PIN default now matches best-effort byte-for-byte
    # -- the ring-branch-ownership fix lives in `name_substituent_fragment`
    # itself (Step 1b, gate-agnostic), so `allow_mancude` no longer changes
    # the answer for this shape. See the module docstring for the full
    # derivation + the RT-safety evidence.
    m, frag, S = _sulfanyl_frag(mol_smi)
    assert name_substituent(m, frag, S, allow_mancude=False) == expected


# A ring-ASSEMBLY-bearing R (-CH2-[1,1'-biphenyl]-4-yl) is named as the proper
# `([1,1'-biphenyl]-4-yl)methyl` (WITH the free-valence -yl), so its -S- form
# `[([1,1'-biphenyl]-4-yl)methyl]sulfanyl` ROUND-TRIPS and the probe re-anchor
# accepts it. (tail: the re-anchor previously rejected EVERY ring R because
# the probe thioether was built with a trivalent `C[SH]...` sulfur -- the S-parent
# bond was cut and RDKit filled the freed valence with an H, so no divalent-S
# name could ever match. The probe now detaches the parent bond first, giving a
# clean divalent CH3-S-R, so a genuinely RT-valid ring R is named and only a name
# that does NOT round-trip -- e.g. a yl-less parent-hydride leak -- still fails
# closed. Whole molecule verified: 1-methyl-4-{[([1,1'-biphenyl]-4-yl)methyl]-
# sulfanyl}benzene round-trips to the input.)
@pytest.mark.opsin_gate
def test_ring_assembly_R_names_when_it_round_trips():
    import orthonym.assembly.substituent_enumerator as SE
    m, frag, S = _sulfanyl_frag("Cc1ccc(SCc2ccc(-c3ccccc3)cc2)cc1")
    assert (SE._name_thio_ring_branch(m, set(frag), S)
            == "[([1,1'-biphenyl]-4-yl)methyl]sulfanyl")


# --- tail #23: the -S-X sulfenyl-halide substituent -> {halo}sulfanyl. ---
# Tier 1.97 named -S-R only when the S continuation was carbon; a halogen fell
# through, so decalin-SCl abstained. Extend to a terminal-halogen continuation.
def _schalide_frag(mol_smi):
    """(mol, frag_atoms={S,X}, attach_S) for a ring/chain -S-X substituent."""
    m = Chem.MolFromSmiles(mol_smi)
    assert m is not None, mol_smi
    S = next(a.GetIdx() for a in m.GetAtoms()
             if a.GetSymbol() == 'S' and a.GetDegree() == 2
             and a.GetFormalCharge() == 0
             and any(nb.GetSymbol() in ('F', 'Cl', 'Br', 'I')
                     for nb in a.GetNeighbors()))
    X = next(nb.GetIdx() for nb in m.GetAtomWithIdx(S).GetNeighbors()
             if nb.GetSymbol() in ('F', 'Cl', 'Br', 'I'))
    return m, sorted([S, X]), S


SCHALIDE = [
    ("ClSC1CCCCC1", "chlorosulfanyl"),
    ("ClSCC", "chlorosulfanyl"),
    ("BrSC1CCCCC1", "bromosulfanyl"),
]


@pytest.mark.parametrize("mol_smi,expected", SCHALIDE)
def test_best_effort_names_sulfenyl_halide(mol_smi, expected):
    m, frag, S = _schalide_frag(mol_smi)
    assert name_substituent(m, frag, S, allow_mancude=True) == expected


@pytest.mark.parametrize("mol_smi,_expected", SCHALIDE)
def test_sulfenyl_halide_pin_default_byte_identical(mol_smi, _expected):
    # best-effort-only scope: PIN default keeps the historical sentinel.
    m, frag, S = _schalide_frag(mol_smi)
    assert name_substituent(m, frag, S, allow_mancude=False) == "substituent"


@pytest.mark.opsin_gate
def test_decahydronaphthalene_sulfenyl_chloride_emits():
    """#23 whole molecule round-trips under best-effort."""
    from orthonym.namer import Orthonym
    r = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True).name_tiered("C1CCC2C(C1)CCCC2SCl")
    assert r.get("name")  # emits (not abstain)
    assert "chlorosulfanyl" in r["name"]

