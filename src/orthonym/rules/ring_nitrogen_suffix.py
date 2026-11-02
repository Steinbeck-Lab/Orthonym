"""A suffix on the ring nitrogen of an azine is cited with added hydrogen.

 "'Added indicated hydrogen'" (the Blue Book) is used "to introduce... a principal characteristic
group into a fully unsaturated heteromonocyclic compound or fused polycyclic system in the absence of, or lack of,
sufficient hydrogen atoms to accommodate the operation at the site of the operation... or on equivalent heteroatoms
such as -N=". The nitrogen of pyridine and quinoline has no hydrogen to give, so the suffix on it takes added
hydrogen and 'hydro' prefixes cover the other saturated positions: 'quinoline-1(2H)-carboxylic acid (PIN)'
(:24701, and:29894), 'pyridin-1(2H)-yl (preferred prefix)' (:24703), and not '1,2-dihydropyridine-1-ide'
but 'pyridin-1(2H)-ide (PIN)',:40928).

Scope (fail-closed): one neutral fragment without stereo whose only ring system is a six-membered azine or a
benzo-fused azine of the table below, one of whose nitrogens carries a carboxy, carbamoyl or formyl group, and
nothing else. The hydrogen is placed by the shared rule (``rules.ring_hydrogen``); the numbering is the parent's,
then the lowest locants for the suffix, the added hydrogen and the hydro prefixes.
"""
from typing import Dict, Optional, Tuple

from rdkit import Chem

#: parent name -> (aromatic SMILES, {atom index of that SMILES: locant})
_PARENTS: Dict[str, Tuple[str, Dict[int, object]]] = {
    'pyridine': ('c1ccncc1', {3: 1, 4: 2, 5: 3, 0: 4, 1: 5, 2: 6}),
    'pyrimidine': ('c1cncnc1', {2: 1, 3: 2, 4: 3, 5: 4, 0: 5, 1: 6}),
    'pyrazine': ('c1cnccn1', {2: 1, 3: 2, 4: 3, 5: 4, 0: 5, 1: 6}),
    'pyridazine': ('c1ccnnc1', {3: 1, 4: 2, 5: 3, 0: 4, 1: 5, 2: 6}),
}
#: the benzo-fused azines of ``data.fused_heterocycles`` (the same numbering)
_FUSED = ('quinoline', 'isoquinoline', 'quinazoline', 'quinoxaline', 'cinnoline', 'phthalazine')


def _parents():
    from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    out = dict(_PARENTS)
    for smi, entry in FUSED_HETEROCYCLE_DATA.items():
        if entry.get('name') in _FUSED and entry.get('iupac_locants'):
            out[entry['name']] = (smi, entry['iupac_locants'])
    return out


def _suffix(mol, n_idx: int, ring) -> Optional[Tuple[str, set]]:
    """The suffix of the group on ring nitrogen ``n_idx`` and its atoms, or None."""
    out = [nb for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors() if nb.GetIdx() not in ring]
    if len(out) != 1 or out[0].GetSymbol() != 'C' or out[0].GetIsAromatic():
        return None
    c = out[0]
    rest = [nb for nb in c.GetNeighbors() if nb.GetIdx() != n_idx]
    oxo = [nb for nb in rest if nb.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2.0]
    if len(oxo) != 1 or oxo[0].GetDegree() != 1:
        return None
    other = [nb for nb in rest if nb.GetIdx() != oxo[0].GetIdx()]
    atoms = {c.GetIdx(), oxo[0].GetIdx()}
    if not other:
        return ('carbaldehyde', atoms) if c.GetTotalNumHs() == 1 else None
    if len(other) != 1 or other[0].GetDegree() != 1:
        return None
    x = other[0]
    atoms.add(x.GetIdx())
    if x.GetSymbol() == 'O' and x.GetTotalNumHs() == 1:
        return 'carboxylic acid', atoms
    if x.GetSymbol() == 'N' and x.GetTotalNumHs() == 2:
        return 'carboxamide', atoms
    return None


def name_azine_nitrogen_suffix(mol) -> Optional[str]:
    """The name of ``mol`` when a suffix on the ring nitrogen of an azine needs added hydrogen, else None."""
    from . import ring_hydrogen
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    for a in mol.GetAtoms():
        if a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return None
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()):
        return None
    ring = frozenset(i for r in mol.GetRingInfo().AtomRings() for i in r)
    if not ring or any(mol.GetAtomWithIdx(i).GetSymbol() not in ('C', 'N') for i in ring):
        return None
    sites = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == 'N'
             and sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in ring) == 1]
    if len(sites) != 1:
        return None
    n_idx = sites[0]
    found = _suffix(mol, n_idx, ring)
    if found is None:
        return None
    suffix, suffix_atoms = found
    if ring | suffix_atoms != frozenset(range(mol.GetNumAtoms())):
        return None                       # another group on the molecule: not built here
    km = Chem.RWMol(mol)
    try:
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001 -- a structure RDKit cannot kekulize is not built here
        return None
    if not any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring
               for b in km.GetBonds()):
        return None                       # a saturated ring has a name of its own ('piperidine')
    best = None
    for parent, (smi, locs) in _parents().items():
        q = Chem.MolFromSmiles(smi)
        if q is None or q.GetNumAtoms() != len(ring):
            continue
        Chem.Kekulize(q, clearAromaticFlags=True)
        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeBondsGeneric = True
        q = Chem.AdjustQueryProperties(q, params)
        for match in km.GetSubstructMatches(q, uniquify=False):
            if frozenset(match) != ring:
                continue
            a2l = {match[i]: locs[i] for i in range(len(match))}
            rh = ring_hydrogen.ring_hydrogen(mol, ring, {n_idx}, a2l)
            if rh is None or not rh.added:
                continue
            key = ring_hydrogen.numbering_key(rh, a2l, {n_idx})
            if best is None or key < best[0]:
                best = (key, parent, a2l, rh)
    if best is None:
        return None
    _key, parent, a2l, rh = best
    front = ring_hydrogen.parent_prefix(rh, a2l)
    if front is None:
        return None
    return f"{front}{parent}-{a2l[n_idx]}({ring_hydrogen.added_text(rh, a2l)})-{suffix}"
