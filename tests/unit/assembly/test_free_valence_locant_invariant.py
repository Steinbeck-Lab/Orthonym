"""Task A (v29 residue): the free valence is a STRUCTURAL fact, never a borrowed one.

The Step-3/4 "cap the free valence with H -> name the capped molecule -> string-surgery
the suffix into a prefix" path hands ``parent_to_prefix`` two things only: a *name string*
and a *carbon count*. Neither carries the attachment position, so every locant the
converter splices into the prefix is borrowed from the CAPPED molecule's own numbering.

That numbering is chosen to favour the capped molecule's principal characteristic group,
which is exactly the numbering P-46.1.8 forbids for a substituent:

    P-46.1 criterion (h) / P-46.1.8 (BlueBookV2.md): "The principal substituent chain has
    the lowest locants for free valences of any kind."

    P-29.2 "GENERAL METHODOLOGY FOR NAMING SUBSTITUENT GROUPS", method (2): "The locants
    for the atoms of free valences are as low as is consistent with any established
    numbering of the parent hydride and, except for mononuclear parent hydrides or the
    suffix 'ylidyne', the locant '1' must be cited."

Worked witness (R8.2). Fragment ``-C(CH3)(C2H5)-(CH2)8-CH(NH2)-CH(CH3)2``:

  * capped and named as a free molecule -> ``2,12-dimethyltetradecan-3-amine``
    (numbered so the AMINE gets locant 3), string-surgered to
    ``3-amino-2,12-dimethyltetradecyl`` -- no free-valence locant at all;
  * numbered structurally from the free valence -> ``12-amino-3,13-dimethyltetradecan-3-yl``.

The two locant SETS differ (amino 3 vs 12, methyls 2,12 vs 3,13) because the chain is
numbered from opposite ends. That is why merely *reading* the dead ``attach_locant``
parameter and splicing it onto the borrowed stem cannot work: it would yield
``3-amino-2,12-dimethyltetradecan-3-yl``, which mixes two incompatible numberings.
The whole numbering has to be recomputed from the structure.

So this file pins two invariants:

  1. STRUCTURAL -- an acyclic fragment whose only heteroatoms are simple detachable
     branches is numbered from its own free valence, amino included.
  2. FAIL-CLOSED -- ``parent_to_prefix`` may not emit a prefix carrying a locant it
     cannot justify. The attach locant is a REQUIRED argument, and when the caller
     cannot prove it, the locant-splicing branches decline instead of fabricating.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import (
    ATTACH_LOCANT_UNKNOWN,
    _located_acyclic_alkyl_name,
    name_substituent_fragment,
    parent_to_prefix,
)


def _fragment_off_ring(smiles):
    """Split a molecule into (mol, sub_atoms, attach_idx) for the single acyclic
    substituent hanging off a ring. Structural, so the test does not hard-code
    RDKit atom indices that a SMILES rewrite would silently invalidate."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    ring_info = mol.GetRingInfo()
    ring = {a.GetIdx() for a in mol.GetAtoms() if ring_info.NumAtomRings(a.GetIdx())}
    best = None
    for r in sorted(ring):
        for nbr in mol.GetAtomWithIdx(r).GetNeighbors():
            if nbr.GetIdx() in ring:
                continue
            # flood the acyclic component reachable from this neighbour
            seen, stack = {r}, [nbr.GetIdx()]
            frag = []
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                frag.append(cur)
                for nn in mol.GetAtomWithIdx(cur).GetNeighbors():
                    if nn.GetIdx() not in seen:
                        stack.append(nn.GetIdx())
            if best is None or len(frag) > len(best[1]):
                best = (mol, frag, nbr.GetIdx())
    assert best is not None, f"no acyclic substituent found in {smiles}"
    return best


# --------------------------------------------------------------------------
# Invariant 1: the free valence is numbered structurally (amino admitted)
# --------------------------------------------------------------------------

# The R8.2 fragment, hung off a benzene so the substituent namer is the unit
# under test rather than the quinone ring engine.
R82_PROBE = "CCC(C)(CCCCCCCCC(C(C)C)N)c1ccccc1"


def test_amino_branched_alkyl_is_numbered_from_the_free_valence():
    """P-46.1.8: the free valence takes the lowest locant, so the chain is
    numbered from the attachment end and the amine follows -- not the reverse."""
    mol, sub_atoms, attach = _fragment_off_ring(R82_PROBE)
    got = _located_acyclic_alkyl_name(mol, sub_atoms, attach)
    assert got is not None, (
        "the structural namer declined an acyclic fragment whose only heteroatom "
        "is a terminal -NH2; it therefore falls through to cap-and-rename"
    )
    name, k = got
    assert name == "12-amino-3,13-dimethyltetradecan-3-yl"
    assert k == 3


def test_amino_branched_alkyl_via_the_fragment_namer():
    """The same fragment through the public entry point."""
    mol, sub_atoms, attach = _fragment_off_ring(R82_PROBE)
    assert name_substituent_fragment(mol, sub_atoms, attach, []) == (
        "12-amino-3,13-dimethyltetradecan-3-yl"
    )


def test_secondary_amino_branch_still_declines_fail_closed():
    """Proof, not deny-list: an -NH-CH3 branch is NOT a simple detachable
    'amino' prefix, so the shape proof must still refuse it rather than emit
    'amino' and drop the methyl."""
    mol, sub_atoms, attach = _fragment_off_ring("CCC(C)(CCCCNC)c1ccccc1")
    got = _located_acyclic_alkyl_name(mol, sub_atoms, attach)
    assert got is None


def test_pure_alkyl_numbering_is_unchanged():
    """Regression anchor: admitting nitrogen must not perturb the all-carbon
    forms this deriver already produced."""
    mol, sub_atoms, attach = _fragment_off_ring("CCC(C)c1ccccc1")
    got = _located_acyclic_alkyl_name(mol, sub_atoms, attach)
    assert got is not None
    assert got[0] == "butan-2-yl"


# --------------------------------------------------------------------------
# Invariant 2: parent_to_prefix may not fabricate a free-valence locant
# --------------------------------------------------------------------------

def test_attach_locant_is_a_required_argument():
    """'Make a prefix unrenderable without its locant' (:345).
    Omitting the attach locant must be impossible, not silently default to 1."""
    with pytest.raises(TypeError):
        parent_to_prefix("propan-2-ol", 3)          # positional-only call
    with pytest.raises(TypeError):
        parent_to_prefix("propan-2-ol", chain_length=3)


@pytest.mark.parametrize("parent,n", [
    # every branch that splices a locant borrowed from the CAPPED molecule's
    # numbering into the prefix
    ("propan-2-ol", 3),                              # -N-ol
    ("(3S)-oct-1-en-3-ol", 8),                       # -N-ol, the prostaglandin class
    ("2,12-dimethyltetradecan-3-amine", 16),         # -an-N-amine, R8.2
    ("butan-2-one", 4),                              # -an-N-one
    ("propane-1,2-diol", 3),                         # -diol
    ("hexane-2,5-dione", 6),                         # -dione
    ("ethane-1,2-diamine", 2),                       # -diamine
    ("2-methylpropanal", 4),                         # aldehyde, R12.1's count bug
    ("butanoic acid", 4),                            # -oic acid, count-derived locant
])
def test_located_branches_fail_closed_without_a_proven_attach_locant(parent, n):
    """The locant these branches would emit belongs to a DIFFERENT molecule's
    numbering. With no proven free-valence locant they must decline."""
    assert parent_to_prefix(
        parent, chain_length=n, attach_locant=ATTACH_LOCANT_UNKNOWN) is None


@pytest.mark.parametrize("parent,n,expected", [
    # forms that carry NO free-valence locant and so need no proof
    ("propane", 3, "propyl"),
    ("pyridine", 0, "pyridinyl"),
    ("methanol", 1, "hydroxymethyl"),
    ("cyclohexane", 6, "cyclohexyl"),
    # P-14.3.4.6 gain: a one-position stem drops the locant it could never
    # justify. These were '1-carbamoylmethyl' / '1-cyanomethyl' / '1-oxomethyl'.
    ("acetamide", 2, "carbamoylmethyl"),
    ("acetonitrile", 2, "cyanomethyl"),
    ("methanal", 1, "oxomethyl"),
])
def test_unlocanted_forms_are_unaffected(parent, n, expected):
    assert parent_to_prefix(
        parent, chain_length=n, attach_locant=ATTACH_LOCANT_UNKNOWN) == expected


def test_existing_fail_closed_contract_is_preserved():
    """v26 BP-2 RC-1 must keep holding through the new signature."""
    for bad in ("isothiocyanic acid", "ethyl formate", "prop-2-enal"):
        assert parent_to_prefix(
            bad, chain_length=3, attach_locant=ATTACH_LOCANT_UNKNOWN) is None


def test_no_emitted_prefix_carries_an_unjustified_locant():
    """End-to-end: the prostaglandin side chain used to reach an emitted name as
    '3-hydroxy(3S)-oct-1-enyl' -- a substituted prefix with no attachment locant
    and a missing hyphen. It must no longer be produced."""
    mol, sub_atoms, attach = _fragment_off_ring(
        "CCCCC[C@H](O)/C=C/C1CCCC1"
    )
    got = name_substituent_fragment(mol, sub_atoms, attach, [])
    assert got is None or "oct-1-enyl" not in got, got
