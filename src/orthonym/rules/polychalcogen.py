"""Homogeneous chalcogen-chain parent hydrides (P-21.2.2 / P-68.3).

A chain of *n* IDENTICAL chalcogen atoms (O / S / Se / Te) singly bonded
end-to-end and terminated by H (potential terminal -OH/-SH functionality is
ignored, P-21.2.2) is a preselected parent hydride named ``<multiplier><stem>``::

    OO    -> dioxidane     (H2O2)      OOO  -> trioxidane    (H2O3)
    SS    -> disulfane     (H2S2)      SSSS -> tetrasulfane   (H2S4)
    [SeH][SeH] -> diselane             [TeH][TeH] -> ditellane

When the two TERMINAL chalcogens instead bear a simple organyl group, the chain
is named substitutively with locants — but ONLY for chains of **>=3** chalcogens::

    CSSS  -> 1-methyltrisulfane          CSSSC -> 1,3-dimethyltrisulfane

A 1-chalcogen "chain" (CSC) is a sulfide and a 2-chalcogen one (CSSC) a
disulfide — both named by sulfanyl-ether nomenclature elsewhere, NOT as a
``-sulfane`` (Blue Book compound-class index 36: "polysulfanes … but not
disulfides or sulfides"). The bare H-terminated chains carry no such ambiguity
and are admitted from n>=2.

Graph classifier (NOT SMARTS — feedback_smarts_and_seniority): every guard
narrows. An oxoacid (chalcogen with =O), a sulfoxide/sulfone, a hetero-chain
(O-S), an internal-substituted (non-standard-valence) chalcogen, a ring, an ion
or a radical all fail a guard and cascade onward — zero false positives.
"""
from typing import List, Optional, Tuple

from rdkit import Chem

from .lambda_convention import format_lambda_token, nonstandard_bonding_number
from .substituent_purity import pure_organyl_prefix_name

# Chalcogen element -> parent-hydride stem (P-21.1 / P-21.2.2).
_CHALCOGEN_STEMS = {'O': 'oxidane', 'S': 'sulfane', 'Se': 'selane', 'Te': 'tellane'}

# Basic multiplying prefixes (Table 1.4). NO elision of the terminal vowel
# (P-21.2.2): di+oxidane -> "dioxidane", tetra+oxidane -> "tetraoxidane".
_MULTIPLIER = {
    2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa', 7: 'hepta', 8: 'octa',
}
# Substituent multiplying prefixes (used WITH locants, normal elision rules).
_SUB_MULTIPLIER = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}


def _chalcogen_chain(mol, element: str) -> Optional[List[int]]:
    """Return the chalcogen atom indices ordered as a simple linear path, or None
    if the chalcogens of ``element`` do not form one (branch, ring, fork)."""
    chal = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == element]
    chal_set = set(chal)
    # Each chalcogen's chalcogen-neighbours via SINGLE bonds; build adjacency.
    adj = {i: [] for i in chal}
    for i in chal:
        atom = mol.GetAtomWithIdx(i)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j in chal_set:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                    return None
                adj[i].append(j)
    # Linear path: exactly two endpoints (degree 1), the rest degree 2.
    degrees = {i: len(adj[i]) for i in chal}
    if any(d > 2 for d in degrees.values()):
        return None
    endpoints = [i for i, d in degrees.items() if d == 1]
    if len(chal) == 1:
        return chal                       # single chalcogen (sulfide hub) handled by caller scope
    if len(endpoints) != 2:
        return None                       # ring (no endpoint) or fork
    # Walk from one endpoint to the other.
    order = [endpoints[0]]
    prev = -1
    cur = endpoints[0]
    while True:
        nxts = [j for j in adj[cur] if j != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    if len(order) != len(chal):
        return None
    return order


def _terminal_substituents(mol, chain: List[int]
                           ) -> Optional[List[Tuple[int, str]]]:
    """Validate every chalcogen's non-chain environment and return the list of
    ``(chain_index_0_based, organyl_name)`` substituents, or None on any violation.

    Internal chalcogens must carry only H (a substituent there is non-standard
    valence). Terminal chalcogens carry exactly one H OR one pure organyl."""
    chain_set = set(chain)
    subs: List[Tuple[int, str]] = []
    for pos, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        heavy_nonchain = [n for n in atom.GetNeighbors()
                          if n.GetIdx() not in chain_set and n.GetSymbol() != 'H']
        is_terminal = pos in (0, len(chain) - 1)
        if not heavy_nonchain:
            continue
        if not is_terminal or len(heavy_nonchain) != 1:
            return None                   # internal substituent / >1 organyl
        name = pure_organyl_prefix_name(mol, heavy_nonchain[0].GetIdx(), idx)
        if name is None:
            return None
        subs.append((pos, name))
    return subs


def _format_substituents(subs: List[Tuple[int, str]], n: int) -> str:
    """Build the locant + alphabetised multiplied substituent prefix string,
    choosing the chain orientation that gives the lowest locant set."""
    # Two orientations: position p or (n-1-p). Pick the lower locant multiset.
    def locants(flip):
        return sorted((n - 1 - p if flip else p) + 1 for p, _ in subs)
    forward, reverse = locants(False), locants(True)
    flip = reverse < forward
    placed = [((n - 1 - p if flip else p) + 1, name) for p, name in subs]
    # Group identical substituents -> "1,3-dimethyl"; alphabetise by name.
    by_name: dict = {}
    for loc, name in placed:
        by_name.setdefault(name, []).append(loc)
    parts = []
    for name in sorted(by_name):
        locs = sorted(by_name[name])
        mult = _SUB_MULTIPLIER.get(len(locs), '')
        parts.append((min(locs), f"{','.join(str(l) for l in locs)}-{mult}{name}"))
    parts.sort()  # cite in ascending first-locant order (matches alpha here)
    return ''.join(p[1] for p in parts)


def name_chalcogen_chain(mol) -> Optional[str]:
    """Return the PIN for a homogeneous chalcogen-chain parent hydride (P-21.2.2),
    else None (fail-closed cascade-continuation). Pure: no mol mutation."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    # Exactly one chalcogen element present (homogeneous chain).
    chalcogens = {a.GetSymbol() for a in mol.GetAtoms()
                  if a.GetSymbol() in _CHALCOGEN_STEMS}
    if len(chalcogens) != 1:
        return None
    element = next(iter(chalcogens))

    # Every heavy atom is either the chalcogen or a carbon (organyl). Any other
    # heteroatom (N, P, halogen, a second chalcogen) -> decline.
    if any(a.GetSymbol() not in (element, 'C')
           for a in mol.GetAtoms() if a.GetSymbol() != 'H'):
        return None

    chain = _chalcogen_chain(mol, element)
    if chain is None or len(chain) < 2:
        return None
    n = len(chain)

    # P-21.2.4: an INTERNAL chalcogen may carry a nonstandard bonding number
    # filled entirely by H (2λ6,5λ4-hexasulfane, 2λ4-trisulfane — both BB
    # verbatim preselected names). A λ terminal, an organyl/heteroatom on the
    # λ atom, or a valence the shared table cannot certify fails closed.
    lam_by_pos = {}
    for pos, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetTotalValence() == 2:
            continue
        if pos in (0, len(chain) - 1):
            return None
        # _terminal_substituents below already refuses internal heavy
        # substituents; the extra valences here are H by construction.
        lam = nonstandard_bonding_number(mol, idx)
        if lam is None:
            return None
        lam_by_pos[pos] = lam

    subs = _terminal_substituents(mol, chain)
    if subs is None:
        return None

    stem = _CHALCOGEN_STEMS[element]
    base = f"{_MULTIPLIER[n]}{stem}" if n in _MULTIPLIER else None
    if base is None:
        return None  # chain too long for the basic multiplier table

    if lam_by_pos:
        if subs:
            return None  # substituted λ-chains not built — fail closed
        # P-21.2.4.1: low locants to the λ set; P-21.2.4.2: on a positional
        # tie the HIGHER bonding number takes the lower locant (λ6 before λ4).
        # sorted (locant, -λ) keys implement both tiers lexicographically.
        fwd = sorted((p + 1, -l) for p, l in lam_by_pos.items())
        rev = sorted((n - p, -l) for p, l in lam_by_pos.items())
        chosen = min(fwd, rev)
        prefix = ','.join(format_lambda_token(loc, -neg) for loc, neg in chosen)
        return f"{prefix}-{base}"

    if not subs:
        return base
    # Carbon-substituted chains are admitted only for n>=3 (avoid sulfide/disulfide).
    if n < 3:
        return None
    return f"{_format_substituents(subs, n)}{base}"


__all__ = ["name_chalcogen_chain"]
