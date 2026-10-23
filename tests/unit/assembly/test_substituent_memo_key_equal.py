"""Lever G: the optimised _substituent_memo_key must return EXACTLY the value of the frozen
reference below (a verbatim copy of the function as of commit b48165466, 2026-09-12), followed by
the three breadth-flag components that dd759cc50 (a lever, 2026-09-17) appended on purpose and the
engine-ordering mark that engine_prefix_ordering appended after them."""
import itertools
import pytest
from rdkit import Chem
from orthonym.assembly import substituent_enumerator as se
from orthonym.assembly import memo
from orthonym.metrics.provenance import (
    allow_aromatic_general_ctx,
    full_coverage_ctx,
    general_fallback_ctx,
)
from orthonym.perception.molcache import bonds_of


# ---- frozen reference (verbatim copy; do not edit) -------------------------------------------
def _reference_key(mol, frag_atoms, attach_idx, allow_mancude,
                          best_effort_val):
    """Complete cache key for:func:`name_substituent` (M1 Lever C1).

    Every component is a proven wrong-name mechanism if omitted
    (`internal notes` Part 1 Lever C):

    1. rooted canonical isomeric fragment SMILES -- structure, stereo parities,
       charges, isotopes AND attachment position (``pentyl`` vs ``pentan-2-yl``);
    2. attach-in-fragment flag (rooted vs unrooted);
    3. external free-valence order (single vs double attachment -> ``-yl`` vs
       ``-ylidene``,;
    4. effective ``allow_mancude`` + best-effort tier (what Tier-4.5 may return);
    5. per-fragment-atom + per-in-fragment-bond ``_CIPCode`` state, in the
       canonical fragment-SMILES output order, so a difference in whether
       ``assign_stereochemistry`` has run on THIS mol object is a cache MISS,
       never a wrong-descriptor hit.

    Returns ``None`` to SKIP caching (fail-open -- a miss can never corrupt) when
    the canonical atom output-order prop is unavailable; the caller treats a
    raised exception the same way.
    """
    attach_in_frag = attach_idx in frag_atoms
    smi = Chem.MolFragmentToSmiles(
        mol, atomsToUse=sorted(frag_atoms),
        rootedAtAtom=(attach_idx if attach_in_frag else -1),
        isomericSmiles=True, canonical=True)
    frag_set = set(frag_atoms)
    _a = mol.GetAtomWithIdx(attach_idx)
    ext_free_valence = sum(
        int(b.GetBondTypeAsDouble()) for b in _a.GetBonds()
        if b.GetOtherAtom(_a).GetIdx() not in frag_set)
    if not mol.HasProp('_smilesAtomOutputOrder'):
        return None
    _order = [int(x) for x in mol.GetProp('_smilesAtomOutputOrder')
              .strip('[]').replace(' ', '').split(',') if x != '']
    atom_cips = tuple(
        (mol.GetAtomWithIdx(i).GetProp('_CIPCode')
         if mol.GetAtomWithIdx(i).HasProp('_CIPCode') else None)
        for i in _order)
    _rank = {idx: pos for pos, idx in enumerate(_order)}
    _bond_items = []
    for _b in bonds_of(mol):
        _bi, _ei = _b.GetBeginAtomIdx(), _b.GetEndAtomIdx()
        if _bi in frag_set and _ei in frag_set:
            _cip = _b.GetProp('_CIPCode') if _b.HasProp('_CIPCode') else None
            _bond_items.append(((_rank.get(_bi, -1), _rank.get(_ei, -1)), _cip))
    _bond_items.sort(key=lambda t: t[0])
    bond_cips = tuple(c for _, c in _bond_items)
    # Item-1 fix : MolFragmentToSmiles drops each fragment atom's bonds to
    # atoms OUTSIDE the fragment and back-fills implicit H, so two structurally
    # distinct substituents (terminal formyl-on-N `formamido` vs an acyl-bridge
    # `carbamoyl`) collide on the same `smi`. `ext_free_valence` above measures
    # only the ATTACH atom, so a non-attach atom's external bond was invisible.
    # Capture every fragment atom's external-bond-order multiset, in the canonical
    # fragment-SMILES output order, so the key is COMPLETE. Finer keys only turn a
    # false HIT into a MISS (recompute) -> byte-identity-safe.
    ext_bond_orders = tuple(
        tuple(sorted(
            b.GetBondTypeAsDouble()
            for b in mol.GetAtomWithIdx(i).GetBonds()
            if b.GetOtherAtom(mol.GetAtomWithIdx(i)).GetIdx() not in frag_set))
        for i in _order)
    # Item-1 fix, a performance pass (, a review C1): the fragment SMILES + external-bond
    # ORDERS still describe only the fragment and the multiplicities of its
    # outward bonds -- NOT what those bonds lead to. Producers on the cascade read
    # BEYOND the fragment: ``_name_amino_branch`` walks the acyl carbon's external
    # chain and counts its carbons (in ONE molecule, a propanoyl and a butanoyl
    # ``-NH-C(=O)-`` bridge both present the ``{N,C,O}`` sub-fragment), and the
    # ring-cut checks read ``RingInfo`` for rings that close through external atoms
    # (a pyrrolidine C4 vs an acyclic C4 diyl). Those distinct inputs cannot be
    # told apart by any FRAGMENT-only signature.
    #
    # The complete-by-construction fix (mirroring ``_POLYFUNC_MEMO_CACHE``'s
    # ``(frozenset(sub_atoms), attach_idx)`` key, substituent_naming.py:1584): the
    # memo scope is ONE top-level naming call, so within it the producer's input is
    # exactly ``(mol, frag_atoms, attach_idx, flags)``. ``frozenset(frag_atoms)``
    # is a PERFECT fragment discriminator within a molecule -- two different
    # substituents are two different atom-index sets, so the propanoyl and butanoyl
    # ``{N,C,O}`` fragments (distinct positions) get distinct keys -- and raw
    # ``attach_idx`` splits the symmetric same-fragment/different-end case (C1.3,
    # ``butan-1-yl`` vs ``butan-4-yl``). The retained structural bits (``smi``,
    # ``ext_bond_orders``, CIP stamp) still discriminate mol OBJECTS, so a copy of
    # the mol with coinciding indices but a different structure cannot false-hit.
    # ``frozenset(frag_atoms)`` strictly subsumes what a canonical-rank signature
    # did (it separates even symmetric fragments a rank collapses), so the key is
    # only FINER -> a false HIT can become only a MISS (recompute) -> byte-identity
    # preserved.
    return (smi, attach_in_frag, ext_free_valence,
            bool(allow_mancude), best_effort_val, atom_cips, bond_cips,
            ext_bond_orders, frozenset(frag_atoms), attach_idx)

# ------------------------------------------------------------------------------------------------

SMILES = ["CC(C)CC(=O)N[C@@H](C)c1ccccc1", "O=C(N[C@@]12CCC[C@@H]1N(C(=O)CC1CCCC1)CCC2)c1ccoc1",
          "CCOC(=O)CC(=O)N[C@@H]1CCCCCN(C(=O)Cc2ccccc2[S@@](C)=O)C1", "C[C@H](O)[C@@H](N)C(=O)O",
          "CN(C[C@H]1CCN(C(=O)C(C)(C)C#N)C1)C(=O)c1cnc2c(c1)CCS(=O)(=O)C2", "Cc1noc(Cl)c1CC(=O)NC[C@H]1C[C@@](F)(CNC(=O)[C@H](C)C2=CCCC2)C1"]


def _prepared(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.MolFragmentToSmiles(mol, atomsToUse=list(range(mol.GetNumAtoms())), canonical=True)  # sets _smilesAtomOutputOrder
    return mol


@pytest.mark.parametrize("smi", SMILES)
def test_key_identical_to_reference(smi):
    mol = _prepared(smi)
    n = mol.GetNumAtoms()
    tok = memo.push_scope()
    try:
        for attach in range(min(n, 8)):
            others = [i for i in range(n) if i != attach]
            for frag in (others[: max(2, n // 2)], others[n // 3: n // 3 + max(2, n // 2)], [attach] + others[:3]):
                for am, be in itertools.product((False, True), (False, True)):
                    key = se._substituent_memo_key(mol, frag, attach, am, be)
                    ref = _reference_key(mol, frag, attach, am, be)
                    if ref is None:  # the fail-open "do not cache" answer must stay the same
                        assert key is None
                        continue
                    # dd759cc50 (a lever) appended the three breadth flags to the key so a value cached
                    # at one breadth configuration is never served to another; the first ten components
                    # are still byte-identical to the frozen reference.
                    assert key[:10] == ref
                    # engine_prefix_ordering appended the mark of a general-engine run (the engine's
                    # allow_mancude decides the ordering names of the prefixes it numbers): False here
                    assert key[10:] == (general_fallback_ctx.get(), allow_aromatic_general_ctx.get(),
                                        full_coverage_ctx.get(), False)
    finally:
        memo.pop_scope(tok)



# ---- the fragment SMILES of the key is handed the fragment's own bonds --------------------------------------
# Left to find them, RDKit walks every bond of the molecule on each rooted call (1.7 ms for a 10-atom fragment of a
# 404-atom molecule, 0.6 ms with the bonds given). The string, and the output-order properties the key's parts and
# later callers read, must be those of the call without them.
MIXED = SMILES + ["CC(=O)N[C@@H](C)C(=O)N[C@H](CC(C)C)C(=O)O.C/C=C/C(=O)[O-].[Na+]", "C/C=C\\C=C/CC(N)=O"]


@pytest.mark.parametrize("smi", MIXED)
def test_key_smiles_and_output_order_equal_the_unrestricted_call(smi):
    mol = _prepared(smi)
    n = mol.GetNumAtoms()
    many = len(Chem.GetMolFrags(mol)) > 1
    tok = memo.push_scope()
    try:
        for attach in range(min(n, 8)):
            others = [i for i in range(n) if i != attach]
            for frag in ([attach], [attach] + others[:1], others[: max(2, n // 2)], [attach] + others[:5], others):
                for rooted in (True, False):
                    att = attach if rooted else n - 1 if attach != n - 1 else 0
                    in_frag = att in frag
                    key = se._substituent_memo_key(mol, frag, att, False, False)
                    got_order = (mol.GetProp('_smilesAtomOutputOrder'), mol.GetProp('_smilesBondOutputOrder'))
                    root = att if in_frag and not (in_frag and many) else -1
                    want = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(frag), rootedAtAtom=root,
                                                    isomericSmiles=True, canonical=True)
                    want_order = (mol.GetProp('_smilesAtomOutputOrder'), mol.GetProp('_smilesBondOutputOrder'))
                    smi_part = key[0][1] if isinstance(key[0], tuple) else key[0]
                    assert smi_part == want
                    assert got_order == want_order
    finally:
        memo.pop_scope(tok)
