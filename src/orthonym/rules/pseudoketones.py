"""Pseudoketone routing for acyl-on-ring-nitrogen "hidden amides" (Fix 3,
Blue Book P-66.1.3 / P-64.3.2 / P-66.1.4.3).

An acyl group on a *ring-system* nitrogen (or any skeletal heteroatom that is not
an acyclic amine N) is NOT named as an amide: the ring N is a skeletal atom of a
ring parent hydride, so the C=O carbon becomes the principal group as a KETONE
(pseudoketone) and the ring fragment is cited as an N-yl substituent on the
carbonyl carbon::

    CC(=O)N1CCCCC1 -> 1-(piperidin-1-yl)ethan-1-one (P-66.1.3)
    CCC(=O)N1CCCCC1 -> 1-(piperidin-1-yl)propan-1-one
    CC(=S)N1CCCC1 -> 1-(pyrrolidin-1-yl)ethane-1-thione (P-66.1.4.3)

Detection is on the original graph: the amide N must be a RING member and the
carbonyl carbon must be EXOCYCLIC to that ring (so a lactam, whose C=O is itself
in the ring, is excluded and stays a cyclic amide/one). An ordinary acyclic
amide (acetamide, N,N-dimethylacetamide, benzamide) has a non-ring amide N and is
never a pseudoketone.

Scope (returns ``None`` to fall through otherwise — never a wrong name): a single
amide group whose acyl side is a clean carbon chain (no further substituents /
heteroatoms / unsaturation on the acyl carbons). The ring fragment is named via
the existing ring-substituent namer, so substituents ON the ring are handled.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from ..assembly.naming_utils import apply_vowel_elision
from ..data.chain_names import get_chain_prefix
from ..perception.chains import find_longest_carbon_chain
from .ring_substituents import name_ring_system_substituent

# Amide-family principal groups that can be a hidden amide (the N is bonded to
# the acyl C). A primary amide N (2 H) cannot be a ring member, so only the
# secondary/tertiary and chalcogen variants ever qualify, but we accept the whole
# family and let the ring-membership test decide.
_AMIDE_FGS = (
    "primary_amide", "secondary_amide", "tertiary_amide",
    "thioamide", "selenoamide", "telluroamide",
)

# carbonyl chalcogen symbol -> ketone (pseudoketone) suffix per P-66.1.3 / P-66.1.4.3.
_CHALCOGEN_KETONE_SUFFIX = {"O": "one", "S": "thione", "Se": "selone", "Te": "tellone"}
_CHALCOGENS = ("O", "S", "Se", "Te")


def _ring_system_atoms(mol, seed_idx: int) -> Tuple[int, ...]:
    """All atoms of the fused ring system reachable from ``seed_idx`` via ring
    bonds (a single ring for piperidine/pyrrolidine/morpholine)."""
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    system = set()
    frontier = [r for r in rings if seed_idx in r]
    seen_rings = []
    while frontier:
        r = frontier.pop()
        if r in seen_rings:
            continue
        seen_rings.append(r)
        system |= r
        for other in rings:
            if other not in seen_rings and (other & system):
                frontier.append(other)
    return tuple(sorted(system))


def is_hidden_amide(mol, amide_match) -> Optional[Tuple[int, int, str]]:
    """Return ``(carbonyl_C_idx, ring_N_idx, chalcogen_symbol)`` if ``amide_match``
    is an acyl-on-ring-N hidden amide, else ``None``.

    Criteria: the amide N is a ring member; the carbonyl carbon is bonded to that
    N and double-bonded to a chalcogen (O/S/Se/Te) and is itself NOT in a ring
    (exocyclic acyl — excludes lactams)."""
    n_idx = None
    for idx in amide_match:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N":
            n_idx = idx
            break
    if n_idx is None:
        return None
    n_atom = mol.GetAtomWithIdx(n_idx)
    if not n_atom.IsInRing():
        return None

    # carbonyl carbon = a carbon neighbour of N that is double-bonded to a chalcogen
    from rdkit import Chem

    for nbr in n_atom.GetNeighbors():
        if nbr.GetSymbol() != "C":
            continue
        c_idx = nbr.GetIdx()
        chalcogen = None
        for cn in nbr.GetNeighbors():
            if cn.GetSymbol() in _CHALCOGENS:
                bond = mol.GetBondBetweenAtoms(c_idx, cn.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    chalcogen = cn.GetSymbol()
                    break
        if chalcogen is None:
            continue
        if mol.GetAtomWithIdx(c_idx).IsInRing():
            return None  # lactam: carbonyl C in the ring -> not a pseudoketone
        return (c_idx, n_idx, chalcogen)
    return None


def name_pseudoketone(features: Any, style: str = "pin") -> Optional[str]:
    """Name an acyl-on-ring-N hidden amide as a pseudoketone, or ``None`` to fall
    through when the structure is outside the clean-acyl-chain class."""
    mol = features.mol
    pg = features.principal_group
    matches = features.principal_group_atoms
    if pg not in _AMIDE_FGS or not matches:
        return None
    if len(matches) != 1:
        return None  # multiple amide groups -> not the simple pseudoketone case

    parsed = is_hidden_amide(mol, matches[0])
    if parsed is None:
        return None
    carbonyl_c, ring_n, chalcogen = parsed
    ketone_suffix = _CHALCOGEN_KETONE_SUFFIX.get(chalcogen)
    if ketone_suffix is None:
        return None

    ring_atoms = _ring_system_atoms(mol, ring_n)
    if not ring_atoms:
        return None

    # The N-side fragment = the whole connected component on the nitrogen side
    # after cutting the N->carbonyl bond (ring system + any substituents on it).
    from collections import deque

    frag_seen = {carbonyl_c}
    frag = []
    queue = deque([ring_n])
    while queue:
        a = queue.popleft()
        if a in frag_seen:
            continue
        frag_seen.add(a)
        frag.append(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() not in frag_seen:
                queue.append(nb.GetIdx())

    # Deterministic constitutional safety guard (no external tool): only emit when
    # the N-side fragment is the BARE ring system (no exocyclic substituents on the
    # ring). For an unsubstituted ring, name_ring_system_substituent reliably yields
    # a constitutionally-correct N-yl name (piperidin-1-yl / pyrrolidin-1-yl /
    # morpholin-4-yl). On a SUBSTITUTED symmetric ring it can drop the attachment
    # locant (e.g. '1-methylpiperazinyl', which denotes a DIFFERENT molecule), so a
    # decorated ring falls through to the legacy path rather than risk a wrong name.
    # (Substituted-ring pseudoketones are a follow-on, gated on the ring-substituent
    # namer gaining correct attachment numbering for symmetric N-heterocycles.)
    if set(frag) != set(ring_atoms):
        return None

    # Acyl parent chain: longest carbon chain that excludes the entire N-side
    # fragment (ring + its substituents); the carbonyl carbon (the ketone, locant
    # 1) must be one terminus of it.
    frag_set = set(frag)
    acyl_chain = find_longest_carbon_chain(mol, exclude_atoms=frag_set)
    if not acyl_chain or carbonyl_c not in acyl_chain:
        return None
    if acyl_chain[0] != carbonyl_c and acyl_chain[-1] != carbonyl_c:
        return None  # carbonyl C must be a chain terminus to be position 1
    chain = acyl_chain if acyl_chain[0] == carbonyl_c else list(reversed(acyl_chain))
    chain_set = set(chain)

    # Clean-acyl guard: the only non-chain heavy neighbours allowed are the
    # carbonyl chalcogen and the ring N (both on the carbonyl carbon). Any other
    # substituent / heteroatom on an acyl carbon -> fall through (not our scope).
    for c in chain:
        catom = mol.GetAtomWithIdx(c)
        for nbr in catom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in chain_set:
                continue
            if nbr.GetSymbol() == "H":
                continue
            if c == carbonyl_c and (ni == ring_n or nbr.GetSymbol() in _CHALCOGENS):
                continue
            return None

    ringyl = name_ring_system_substituent(mol, frag, ring_n)
    if not ringyl or " " in ringyl:
        return None

    length = len(chain)
    base = f"{get_chain_prefix(length)}ane"  # 'ethane', 'propane',...
    joined = apply_vowel_elision(base, ketone_suffix)  # 'ethanone' / 'ethanethione'
    stem_part = joined[: len(joined) - len(ketone_suffix)]  # 'ethan' / 'ethane'
    parent = f"{stem_part}-1-{ketone_suffix}"  # 'ethan-1-one' / 'ethane-1-thione'

    return f"1-({ringyl}){parent}"


__all__ = ["is_hidden_amide", "name_pseudoketone"]


def name_acyl_hetero_pseudoketone(mol) -> Optional[str]:
    """P-64.1.2.1(b) / P-64.5.2.2 acyl on Si/Ge/P/As (Wave-2 completion C):
    CC(=O)[SiH3] -> 1-silylethan-1-one, [PH2]C(=O)CCC -> 1-phosphanylbutan-1-one
    (both BB verbatim). The ketone SMARTS requires C on both flanks, so these
    molecules previously perceived NO functional group and died unnamed.

    Fail-closed graph classifier (mirrors name_pseudoketone): exactly one
    non-ring carbonyl C whose heavy neighbours are {terminal =O, one chain C,
    one acyclic hub in Si/Ge/P/As}; the acyl chain is clean unbranched all-C;
    the hub carries only H (silyl/phosphanyl) or pure organyls
    (trimethylsilyl/dimethylphosphanyl); nothing else in the molecule."""
    from rdkit import Chem
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    _HUBS = {'Si', 'Ge', 'P', 'As'}
    # Group-16 hubs (P-65.6.3.4.2): an acyl on a chalcogen whose non-acyl
    # neighbour is a HETEROATOM (compound substituent) is a pseudoketone
    # (R-CO-S-OO-CH3 -> [1-(methylperoxy)sulfanyl]butan-1-one). A chalcogen hub
    # carrying a PLAIN CARBON (R-CO-S-C) is an ordinary thioester (senior, named
    # elsewhere) and the discriminator below declines it.
    _CHALCOGEN_HUBS = {'S', 'Se', 'Te'}
    _ALL_HUBS = _HUBS | _CHALCOGEN_HUBS

    carbonyls = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C' or atom.IsInRing():
            continue
        oxo = hub = chain_c = None
        extra = False
        for b in atom.GetBonds():
            other = b.GetOtherAtom(atom)
            if (other.GetSymbol() == 'O' and other.GetDegree() == 1
                    and b.GetBondType() == Chem.BondType.DOUBLE):
                oxo = other
            elif (other.GetSymbol() in _ALL_HUBS
                    and b.GetBondType() == Chem.BondType.SINGLE
                    and not other.GetIsAromatic() and not other.IsInRing()):
                hub = other
            elif (other.GetSymbol() == 'C'
                    and b.GetBondType() == Chem.BondType.SINGLE):
                chain_c = other
            else:
                extra = True
        if oxo is not None and hub is not None and chain_c is not None \
                and not extra and atom.GetTotalNumHs() == 0:
            carbonyls.append((atom, oxo, hub, chain_c))
    if len(carbonyls) != 1:
        return None
    catom, oxo, hub, chain_c = carbonyls[0]

    # Hub fragment: the hub + everything on its far side.
    from collections import deque
    hub_frag = []
    seen = {catom.GetIdx()}
    queue = deque([hub.GetIdx()])
    while queue:
        a = queue.popleft()
        if a in seen:
            continue
        seen.add(a)
        hub_frag.append(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() not in seen:
                queue.append(nb.GetIdx())

    # Acyl chain: longest all-C chain excluding the hub side; carbonyl C at
    # a terminus; clean (no other substituents/heteroatoms).
    chain = find_longest_carbon_chain(mol, exclude_atoms=set(hub_frag))
    if not chain or catom.GetIdx() not in chain:
        return None
    if chain[0] != catom.GetIdx() and chain[-1] != catom.GetIdx():
        return None
    if chain[0] != catom.GetIdx():
        chain = list(reversed(chain))
    chain_set = set(chain)
    for c in chain:
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in chain_set:
                continue
            if c == catom.GetIdx() and ni in (oxo.GetIdx(), hub.GetIdx()):
                continue
            return None

    # Group-16 (chalcogen) hub: acyl-hetero pseudoketone (P-65.6.3.4.2). The
    # hub's non-acyl neighbour must be a HETEROATOM (a compound substituent such
    # as the -OO-CH3 peroxy chain); a PLAIN-CARBON non-acyl neighbour makes this
    # an ordinary thioester (R-CO-S-C, senior) which must NOT be swallowed here —
    # decline so the cascade reaches the thioester handler. A bare -SH / -SeH
    # (thioic S-acid) has no heteroatom substituent and also declines.
    if hub.GetSymbol() in _CHALCOGEN_HUBS:
        non_acyl = [nb for nb in hub.GetNeighbors()
                    if nb.GetIdx() != catom.GetIdx()]
        if any(nb.GetAtomicNum() == 6 for nb in non_acyl):
            return None  # thioester (plain carbon on the chalcogen) -> decline
        if not any(nb.GetAtomicNum() not in (1, 6) for nb in non_acyl):
            return None  # no compound-heteroatom substituent -> not this class
        from ..assembly.substituent_enumerator import name_substituent
        hubyl = name_substituent(mol, sorted(hub_frag), hub.GetIdx())
        if not hubyl:
            return None  # unnameable hub substituent -> fail closed
        length = len(chain)
        base_name = f"{get_chain_prefix(length)}an"
        # BB P-65.6.3.4.2 verbatim encloses the located substituent in the outer
        # marks: [1-(methylperoxy)sulfanyl]butan-1-one.
        return f"[1-{hubyl}]{base_name}-1-one"

    # Hub scope (fail-closed): the hub carries only H or pure ORGANYL
    # substituents — a heteroatom on the hub (Si-OH: silanol territory,
    # P-68.2 suffix seniority interplay) is not built here.
    if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6
           for a in hub_frag if a != hub.GetIdx()):
        return None

    # Hub substituent name.
    hubyl = None
    if hub.GetSymbol() in ('Si', 'Ge'):
        from ..assembly.substituent_naming import _name_group14_substituent
        hubyl = _name_group14_substituent(mol, hub_frag, hub.GetIdx())
    else:
        organyls = [n for n in mol.GetAtomWithIdx(hub.GetIdx()).GetNeighbors()
                    if n.GetIdx() != catom.GetIdx() and n.GetAtomicNum() > 1]
        base = 'phosphanyl' if hub.GetSymbol() == 'P' else 'arsanyl'
        if not organyls:
            if len(hub_frag) == 1:
                hubyl = base
        else:
            # P3: the organyl guard is the shared chokepoint, so a hub organyl
            # may now be cyclic, branched, unsaturated or long — and its prefix may
            # carry locants, a retained italicized prefix, or its own marks. The
            # hub is a MONONUCLEAR skeleton, so composition follows P-16.5.1.3.1
            # (BB 7272, verbatim): "the first cited substituent never has enclosing
            # marks unless it includes a locant. The second and further
            # substituents are each enclosed with parentheses even for simple
            # substituents. When the simple substituent groups are accompanied by
            # multiplicative prefixes such as 'di' and 'tri', the multiplicative
            # prefixes are not included in the parentheses." BB 39228
            # `4-[ethyl(methyl)phosphanyl]-1H-imidazole` (PIN) shows the rule
            # holding for this SUBSTITUENT-PREFIX shape (silyl analogue: BB 3545
            # `3-[amino(methyl)silyl]...` (PIN)), which is why the bare-concatenated
            # `ethylmethylphosphanyl` this block used to emit was non-PIN.
            #
            # Every decision below is a shared primitive: citation order
            # (P-14.5.2/P-14.5.4), the compound test (P-16.3.3), the italicized
            # carve-out (P-16.3.4) and the mark escalation (P-16.5.4.1).
            from collections import Counter

            from ..assembly.naming_utils import (apply_enclosing_marks,
                                                 enclose_if_compound,
                                                 multiplied_component,
                                                 prefix_citation_sort_key)
            from .substituent_purity import organyl_prefix_name
            names = []
            for n in organyls:
                nm = organyl_prefix_name(mol, n.GetIdx(), hub.GetIdx())
                if nm is None:
                    return None
                names.append(nm)
            counts = Counter(names)
            # P3-CLOSEOUT Item A: arity bound local, multiplier word from
            # the shared primitive (P-16.3.5(a) bis/tris when SUBSTITUTED).
            _SUPPORTED_COUNTS = frozenset((1, 2, 3))
            parts = []
            for i, nm in enumerate(sorted(counts, key=prefix_citation_sort_key)):
                if counts[nm] not in _SUPPORTED_COUNTS:
                    return None
                marked = enclose_if_compound(nm)
                if i > 0 and marked == nm:
                    marked = apply_enclosing_marks(nm, -1)   # P-16.5.1.3.1
                parts.append(multiplied_component(counts[nm], nm, marked))
            hubyl = ''.join(parts) + base
    if not hubyl:
        return None
    # Compound hub names take enclosing marks (1-(trimethylsilyl)propan-2-one
    # engine precedent); the bare silyl/phosphanyl forms stay unmarked. The marks
    # ESCALATE (-> [ over an inner pair (P-16.5.4.1), which a raw f"({hubyl})"
    # could not do: BB 39228 writes `[ethyl(methyl)phosphanyl]`, never
    # `(ethyl(methyl)phosphanyl)`.
    from ..assembly.naming_utils import apply_enclosing_marks as _marks
    token = hubyl if hubyl in ('silyl', 'germyl', 'phosphanyl', 'arsanyl') \
        else _marks(hubyl, -1)

    length = len(chain)
    base_name = f"{get_chain_prefix(length)}an"
    return f"1-{token}{base_name}-1-one"


# Chalcogen-chain parent-hydride stems (P-21.2.2) for the trioxidanyl substituent.
_CHALCOGENCHAIN_STEMS = {"O": "oxidane", "S": "sulfane", "Se": "selane",
                         "Te": "tellane"}
_CHALCOGENCHAIN_MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}


def name_acyl_chalcogenchain_pseudoketone(mol) -> Optional[str]:
    """P-68.4.1.3 (BB 39389): an acyl group terminating a HOMOGENEOUS chain of
    >=3 identical chalcogens (``-O-O-OH`` = trioxidane) is named as a pseudoketone
    whose carbonyl component is the parent (``-one``) and the chalcogen chain is
    the ``-yl`` substituent::

        CH3-CH2-CO-O-O-OH -> 1-trioxidanylpropan-1-one (PIN)

    GUARD (w3p08 lesson): must NOT swallow ordinary esters/thioesters — an
    ``R-CO-O-C`` ester or ``R-CO-S-C`` thioester has a CARBON on the acyl
    chalcogen, so its "chain" is length 1 (< 3) and is declined; a carboxylic
    acid (-CO-OH, chain length 1) and a peroxy acid (-CO-O-OH, chain length 2)
    likewise decline. Fires only on a pure homogeneous >=3-chalcogen chain
    terminated by -H (the -OH/-SH).

    Fail-closed graph classifier (mirrors name_acyl_hetero_pseudoketone): exactly
    one non-ring carbonyl C whose heavy neighbours are {terminal =O, one chain C,
    one chalcogen hub}; the hub begins a homogeneous n>=3 chalcogen chain (all one
    element, unbranched, H-terminated); the acyl chain is a clean unbranched all-C
    chain; nothing else in the molecule. Pure: no mol mutation.
    """
    from rdkit import Chem
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    carbonyls = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.IsInRing():
            continue
        oxo = hub = chain_c = None
        extra = False
        for b in atom.GetBonds():
            other = b.GetOtherAtom(atom)
            if (other.GetSymbol() == "O" and other.GetDegree() == 1
                    and other.GetTotalNumHs() == 0
                    and b.GetBondType() == Chem.BondType.DOUBLE):
                oxo = other
            elif (other.GetSymbol() in _CHALCOGENCHAIN_STEMS
                    and b.GetBondType() == Chem.BondType.SINGLE
                    and not other.IsInRing()):
                hub = other
            elif (other.GetSymbol() == "C"
                    and b.GetBondType() == Chem.BondType.SINGLE):
                chain_c = other
            else:
                extra = True
        if (oxo is not None and hub is not None and chain_c is not None
                and not extra and atom.GetTotalNumHs() == 0):
            carbonyls.append((atom, oxo, hub, chain_c))
    if len(carbonyls) != 1:
        return None
    catom, oxo, hub, chain_c = carbonyls[0]

    # Hub begins a homogeneous chalcogen chain of >=3 identical chalcogens
    # (unbranched) terminated by H, walking away from the carbonyl C.
    element = hub.GetSymbol()
    chain_idxs = []
    prev, cur = catom.GetIdx(), hub.GetIdx()
    while True:
        a = mol.GetAtomWithIdx(cur)
        if a.GetSymbol() != element or a.GetFormalCharge() != 0:
            return None
        chain_idxs.append(cur)
        nxt = []
        for n in a.GetNeighbors():
            if n.GetIdx() == prev or n.GetAtomicNum() <= 1:
                continue
            if n.GetSymbol() != element:
                return None                       # organyl/heteroatom -> ester guard
            bond = mol.GetBondBetweenAtoms(cur, n.GetIdx())
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            nxt.append(n.GetIdx())
        if len(nxt) == 0:
            break
        if len(nxt) > 1:
            return None                           # branch
        prev, cur = cur, nxt[0]
    if len(chain_idxs) < 3:
        return None                               # ester / peroxy-acid territory
    if mol.GetAtomWithIdx(chain_idxs[-1]).GetTotalNumHs() < 1:
        return None                               # far terminus must be -XH

    # Acyl parent chain: longest all-C chain excluding the chalcogen side; the
    # carbonyl C is a terminus (locant 1); clean (no other substituents).
    hub_frag = set(chain_idxs)
    chain = find_longest_carbon_chain(mol, exclude_atoms=hub_frag)
    if not chain or catom.GetIdx() not in chain:
        return None
    if chain[0] != catom.GetIdx() and chain[-1] != catom.GetIdx():
        return None
    if chain[0] != catom.GetIdx():
        chain = list(reversed(chain))
    chain_set = set(chain)
    for c in chain:
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in chain_set or nbr.GetSymbol() == "H":
                continue
            if c == catom.GetIdx() and ni in (oxo.GetIdx(), hub.GetIdx()):
                continue
            return None

    # Whole molecule = acyl chain + oxo + chalcogen chain (+H).
    covered = chain_set | {oxo.GetIdx()} | hub_frag
    if any(a.GetIdx() not in covered
           for a in mol.GetAtoms() if a.GetAtomicNum() > 1):
        return None

    n = len(chain_idxs)
    stem = _CHALCOGENCHAIN_STEMS[element]
    mult = _CHALCOGENCHAIN_MULT.get(n)
    if mult is None:
        return None
    subst = f"{mult}{stem}"[:-1] + "yl"           # trioxidane -> trioxidanyl
    base = f"{get_chain_prefix(len(chain))}an"
    return f"1-{subst}{base}-1-one"
