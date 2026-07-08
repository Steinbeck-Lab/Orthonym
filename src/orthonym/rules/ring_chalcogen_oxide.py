"""Ring-chalcogen oxide namer (P-25.6 / P-74.3.1.3, Wave-2 completion).

A NEUTRAL ring sulfur/selenium/tellurium bearing 1-2 exocyclic terminal =O
is named additively on the intact ring parent — the S-oxide sibling of the
established ``pyridine 1-oxide`` N-oxide path::

    O=S1c2ccccc2-c2ccccc21        -> dibenzo[b,d]thiophene 5-oxide
    O=S1(=O)c2ccccc2-c2ccccc21    -> dibenzo[b,d]thiophene 5,5-dioxide
    O=S1CCCC1                     -> thiolane 1-oxide

SCOPE (fail-closed, accuracy-first): a single oxidised ring chalcogen; every
non-ring heavy atom of the molecule is one of its oxide oxygens (bare ring
system otherwise — substituted variants cascade onward); the de-oxidised base
ring must resolve BOTH a name and an authoritative IUPAC locant for the
chalcogen. Anything else returns None — never a wrong name.

Graph/atom classifier (no SMARTS broadening); pure — no mol mutation.
"""

from typing import Optional

from rdkit import Chem

_CHALCOGENS = {'S', 'Se', 'Te'}


def name_ring_chalcogen_oxide(mol) -> Optional[str]:
    """Return the additive ring-chalcogen oxide name, else ``None``."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    ri = mol.GetRingInfo()
    if ri.NumRings() == 0:
        return None
    ring_atoms = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)

    # Locate the oxidised ring chalcogen and its terminal =O oxygens.
    site = None
    oxide_oxygens = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in _CHALCOGENS:
            continue
        if atom.GetIdx() not in ring_atoms:
            continue
        oxos = []
        for nb in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            if (nb.GetSymbol() == 'O' and nb.GetDegree() == 1
                    and nb.GetTotalNumHs() == 0
                    and bond.GetBondType() == Chem.BondType.DOUBLE):
                oxos.append(nb.GetIdx())
        if not oxos:
            continue
        if site is not None:
            return None  # two oxidised sites — not built, fail closed
        if len(oxos) > 2:
            return None
        site = atom.GetIdx()
        oxide_oxygens = oxos
    if site is None:
        return None

    # Bare ring system: every non-ring heavy atom must be an oxide oxygen.
    non_ring = {a.GetIdx() for a in mol.GetAtoms()} - ring_atoms
    if non_ring != set(oxide_oxygens):
        return None

    # De-oxidise: remove the =O atoms; the base must sanitize and name.
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIntProp('__rco_orig', a.GetIdx())
    for idx in sorted(oxide_oxygens, reverse=True):
        rw.RemoveAtom(idx)
    base = rw.GetMol()
    try:
        Chem.SanitizeMol(base)
        base_smiles = Chem.MolToSmiles(base, canonical=True)
    except Exception:
        return None

    from ..assembly.fragment_naming import name_fragment_recursively
    try:
        base_name = name_fragment_recursively(base_smiles)
    except Exception:
        return None
    if not base_name or 'unknown' in base_name:
        return None

    # Authoritative locant of the chalcogen under the BASE ring numbering
    # (dibenzothiophene S = 5). Fail closed when the ring-info cascade offers
    # no full iupac_locants map — a guessed locant would be a wrong name.
    base_site = next(
        (i for i in range(base.GetNumAtoms())
         if base.GetAtomWithIdx(i).GetIntProp('__rco_orig') == site), None)
    if base_site is None:
        return None
    try:
        from ..namer import (_build_ring_info_for_parent_selection,
                             compute_features)
        feats = compute_features(base)
        rinfo = _build_ring_info_for_parent_selection(feats)
    except Exception:
        return None
    iupac = (rinfo or {}).get('iupac_locants')
    if not iupac or base_site not in iupac:
        # Single-ring base: the chalcogen is locant 1 by HW convention
        # (thiolane 1-oxide) — accept ONLY the unambiguous one-heteroatom
        # monocycle; everything else fails closed.
        base_ri = base.GetRingInfo()
        if base_ri.NumRings() == 1:
            ring = base_ri.AtomRings()[0]
            hetero_in_ring = [i for i in ring
                              if base.GetAtomWithIdx(i).GetAtomicNum()
                              not in (1, 6)]
            if hetero_in_ring == [base_site]:
                loc = 1
            else:
                return None
        else:
            return None
    else:
        loc = iupac[base_site]
        if isinstance(loc, tuple):
            loc = loc[0]
        if not isinstance(loc, int):
            return None

    if len(oxide_oxygens) == 1:
        return f"{base_name} {loc}-oxide"
    return f"{base_name} {loc},{loc}-dioxide"


__all__ = ["name_ring_chalcogen_oxide"]
