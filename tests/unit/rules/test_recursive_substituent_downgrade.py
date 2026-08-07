"""C4 — recursive-complete substituent namer (downgrade-not-refuse under best-effort).

The keystone: a complex acyclic multi-functional fragment that names STANDALONE (its
principal characteristic group as a SUFFIX) currently declines as a SUBSTITUENT, because
as a substituent every characteristic group must demote to a PREFIX and the attach atom
must carry the free valence (`-yl`). Today the recursive path's suffix->prefix step is the
dead `parent_to_prefix` string surgery, so the atoms are dropped.

These tests drive a structure-based, suffix-free, free-valence chain substituent namer,
gated on best-effort (`allow_mancude`), leaving the PIN default byte-identical.

Same-molecule proof (constitution): the emitted prefix P is embedded on benzene as
`(P)benzene`, parsed by OPSIN, and its skeleton InChIKey must equal the phenyl-capped
fragment's. A prefix that names a DIFFERENT constitution fails rather than passes. The
proof helper is validated on a known positive (isopropyl -> cumene) before it is trusted.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym import namer as NM


def _skeleton_key(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return inchi.InchiToInchiKey(inchi.MolToInchi(m)).split("-")[0]
    except Exception:
        return None


def _phenyl_capped_key(mol, frag_atoms, attach_idx):
    """Skeleton InChIKey of the fragment with a phenyl bonded at the free valence."""
    rw = Chem.RWMol()
    idx = {}
    for a in sorted(frag_atoms):
        src = mol.GetAtomWithIdx(a)
        na = Chem.Atom(src.GetAtomicNum())
        na.SetFormalCharge(src.GetFormalCharge())
        idx[a] = rw.AddAtom(na)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx and j in idx:
            rw.AddBond(idx[i], idx[j], b.GetBondType())
    ph = [rw.AddAtom(Chem.Atom(6)) for _ in range(6)]
    for k in range(6):
        rw.GetAtomWithIdx(ph[k]).SetIsAromatic(True)
        rw.AddBond(ph[k], ph[(k + 1) % 6], Chem.BondType.AROMATIC)
    rw.AddBond(idx[attach_idx], ph[0], Chem.BondType.SINGLE)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return _skeleton_key(Chem.MolToSmiles(m))


def _prefix_names_same_constitution(mol, frag_atoms, attach_idx, prefix):
    """OPSIN-parse `(prefix)benzene` and compare skeleton to the phenyl-capped fragment."""
    opsin_smi = NM._validity_gate_name_to_smiles(f"({prefix})benzene")
    if opsin_smi is None:
        opsin_smi = NM._validity_gate_name_to_smiles(f"{prefix}benzene")
    if opsin_smi is None:
        return None  # OPSIN could not parse -> inconclusive (caller decides)
    want = _phenyl_capped_key(mol, frag_atoms, attach_idx)
    got = _skeleton_key(opsin_smi)
    return want is not None and got is not None and want == got


@pytest.fixture(scope="module")
def opsin_proof():
    """Validate the same-molecule proof helper on a known positive, or skip.

    isopropyl -> `(propan-2-yl)benzene` (cumene) must match phenyl-capped propane.
    If OPSIN is unavailable the helper cannot run; skip rather than pass vacuously.
    """
    prop = Chem.MolFromSmiles("CCC")  # attach at C1 (middle) -> propan-2-yl
    ok = _prefix_names_same_constitution(prop, {0, 1, 2}, 1, "propan-2-yl")
    if ok is None:
        pytest.skip("OPSIN unavailable; cannot run the same-molecule proof")
    assert ok is True, "same-molecule proof helper is broken on the isopropyl positive"
    return _prefix_names_same_constitution


# --- UNIT: the fragment substituent namer directly ---------------------------

def test_isoleucinamide_names_as_covering_substituent(opsin_proof):
    # isoleucinamide-like fragment `CC(C)C[C@H](N)C(N)=O`; standalone it is
    # (2S)-2-amino-4-methylpentanamide, but as a substituent (free valence at the
    # terminal methyl, atom 0) it must demote amide+amine to prefixes + `-yl`.
    mol = Chem.MolFromSmiles("CC(C)C[C@H](N)C(N)=O")
    frag = set(range(mol.GetNumAtoms()))
    prefix = name_substituent(mol, frag, 0, allow_mancude=True)
    assert prefix is not None and prefix != "substituent", f"declined: {prefix!r}"
    assert prefix.rstrip(")").endswith(("yl", "ylidene", "ylidyne")), prefix
    assert opsin_proof(mol, frag, 0, prefix) is True, \
        f"prefix {prefix!r} names a DIFFERENT constitution than the fragment"


def test_chain_composer_handles_unsaturated_backbone(opsin_proof):
    # the backbone stem carries the ene/yne locants (via _stem_block), so an
    # unsaturated chain substituent gets a covering name, not a saturated-stem
    # mis-name. `C/C=C/CC(N)=O` attach at atom 0 -> a pent-2-en-...-yl-family name.
    from orthonym.assembly.substituent_enumerator import (
        _recursive_chain_fragment_substituent_name)
    mol = Chem.MolFromSmiles("C/C=C/CC(N)=O")
    frag = set(range(mol.GetNumAtoms()))
    prefix = _recursive_chain_fragment_substituent_name(
        mol, frag, 0, allow_mancude=True)
    assert prefix is not None, "unsaturated backbone deferred"
    assert ("en-" in prefix or "yn-" in prefix), f"no unsaturation locant: {prefix!r}"
    # OPSIN parses `5-amino-5-oxopent-2-en-1-yl`, so the proof is not vacuous.
    assert opsin_proof(mol, frag, 0, prefix) is True, \
        f"prefix {prefix!r} names a DIFFERENT constitution"


def test_chain_composer_requires_carbon_acyclic_attach():
    from orthonym.assembly.substituent_enumerator import (
        _recursive_chain_fragment_substituent_name)
    # ring attach -> the ring composer's job, not this one.
    mol = Chem.MolFromSmiles("c1ccccc1CC(N)C(N)=O")
    frag = set(range(mol.GetNumAtoms()))
    ring_c = next(a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic())
    assert _recursive_chain_fragment_substituent_name(
        mol, frag, ring_c, allow_mancude=True) is None
    # gated: never runs on the PIN default.
    assert _recursive_chain_fragment_substituent_name(
        mol, frag, 0, allow_mancude=False) is None


def test_composed_prefix_has_no_stray_hyphen_before_stem(opsin_proof):
    # spelling: the alkyl stem elides ('...5-oxopentyl', not '...5-oxo-pentyl').
    mol = Chem.MolFromSmiles("CC(C)C[C@H](N)C(N)=O")
    prefix = name_substituent(mol, set(range(mol.GetNumAtoms())), 0,
                              allow_mancude=True)
    assert prefix and "-pentyl" not in prefix and prefix.endswith("pentyl"), prefix


def test_pin_default_unaffected_simple_substituent():
    # regression guard: the shared PIN path must stay byte-identical.
    mol = Chem.MolFromSmiles("CCCC")  # butane; attach at a terminus -> butyl
    assert name_substituent(mol, {0, 1, 2, 3}, 0, allow_mancude=False) == "butyl"
