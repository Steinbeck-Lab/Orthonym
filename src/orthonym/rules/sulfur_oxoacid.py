""": sulfur-oxoacid acyl-oxy / -amino substituent prefixes.

When a sulfur oxoacid group is attached BY OXYGEN (chalcogen) OR NITROGEN to a
compound that also carries a characteristic group senior to the sulfur acid for
citation as principal group (e.g. a carboxylic acid), the sulfur group is cited
as a substituent PREFIX, not as the parent
, ``the Blue Book Blue Book``):

    3-(sulfooxy)propanoic acid (PIN):36488
    3-[(methoxysulfinyl)oxy]propanoic acid (PIN):36490
    3-[(chlorosulfonyl)oxy]propanoic acid (PIN):36492
    3-(sulfamoyloxy)propanoic acid (PIN):36494
    3-[(aminosulfinyl)oxy]propanoic acid (PIN):36500 (NOT sulfinamoyloxy)
    3-[(methoxysulfonyl)amino]propanoic acid (PIN):36502

The generic path names this tail by skeletal ("a") replacement
(``…-1,3-dioxa-2λ6-thiapropyl``) — a valid, round-tripping, but NON-PIN form.

The S-oxoacid acyl group is ``{X}sulfonyl`` (n S=O double bonds = 2) or
``{X}sulfinyl`` (n = 1), where X is the single S ligand that is neither the
linking O/N nor an oxo. Two Blue Book contractions apply / the
substituent-prefix tables at:56857,:56859): ``hydroxysulfonyl`` -> ``sulfo``
and ``aminosulfonyl`` -> ``sulfamoyl``. ``aminosulfinyl`` is NOT contracted
(explicitly ``[not …sulfinamoyloxy…]`` at:36500).

Every function fails closed (returns ``None``) on any S outside the neutral
mono/di-oxo acyl class (charged/radical S, ring S, an unexpected ligand set,
a carbon-R sulfonyl, a plain thioether); the top-level /OPSIN round-trip
gate keeps 0-wrong on whatever is emitted.
"""
from typing import List, Optional

from rdkit import Chem

# X ligand -> substituent-prefix stem ligand set)
_HALOGEN = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}


def _collect_subtree(mol, start: int, exclude: int) -> List[int]:
    """Atom indices reachable from ``start`` without crossing ``exclude``."""
    seen = {start}
    stack = [start]
    while stack:
        i = stack.pop()
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j == exclude or j in seen:
                continue
            seen.add(j)
            stack.append(j)
    return list(seen)


def _ligand_prefix(mol, x_idx: int, s_idx: int) -> Optional[str]:
    """Name the single non-oxo, non-linker S ligand X as a substituent stem.

    ``-OH`` -> ``hydroxy``, ``-Cl`` -> ``chloro``, ``-O-CH3`` -> ``methoxy``
    (alkoxy via the ordinary substituent cascade), ``-NH2`` -> ``amino``.
    Returns ``None`` (fail closed) for anything else (a carbon-R sulfonyl, a
    substituted amino, a charged ligand,...).
    """
    a = mol.GetAtomWithIdx(x_idx)
    if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons():
        return None
    z = a.GetAtomicNum()
    if z in _HALOGEN and a.GetDegree() == 1:
        return _HALOGEN[z]
    if a.GetSymbol() == "O":
        if a.GetDegree() == 1 and a.GetTotalNumHs() == 1:
            return "hydroxy"
        if a.GetTotalNumHs() == 0:
            others = [n.GetIdx() for n in a.GetNeighbors()
                      if n.GetIdx() != s_idx]
            if len(others) == 1 and mol.GetAtomWithIdx(others[0]).GetSymbol() == "C":
                # -O-alkyl: name the O-alkyl subtree as an oxy prefix
                # (methoxy / ethoxy / …) via the ordinary cascade.
                from ..assembly.substituent_enumerator import name_substituent
                sub = _collect_subtree(mol, x_idx, s_idx)
                nm = name_substituent(mol, sub, x_idx, True)
                if (nm and nm.endswith("oxy")
                        and "(" not in nm and " " not in nm):
                    return nm
        return None
    if a.GetSymbol() == "N":
        if a.GetDegree() == 1 and a.GetTotalNumHs() == 2:
            return "amino"
        return None
    return None


def _sulfur_oxoacid_acyl(mol, s_idx: int, link_idx: int) -> Optional[str]:
    """Return the S-oxoacid acyl prefix for an S bonded to ``link_idx``.

    ``link_idx`` (the O or N linker) is excluded from the ligand walk. The S
    must be neutral, non-radical, acyclic, carry exactly ``n in (1, 2)`` terminal
    ``=O`` groups and exactly ONE further single-bonded ligand X. Returns e.g.
    ``sulfo`` / ``chlorosulfonyl`` / ``sulfamoyl`` / ``aminosulfinyl`` /
    ``methoxysulfonyl`` / ``methoxysulfinyl``; ``None`` if out of class.
    """
    s = mol.GetAtomWithIdx(s_idx)
    if (s.GetSymbol() != "S" or s.GetFormalCharge() != 0
            or s.GetNumRadicalElectrons() or s.IsInRing()):
        return None
    n_oxo = 0
    x_ligands: List[int] = []
    for nb in s.GetNeighbors():
        if nb.GetIdx() == link_idx:
            continue
        bond = mol.GetBondBetweenAtoms(s_idx, nb.GetIdx())
        bt = bond.GetBondType()
        if (nb.GetSymbol() == "O" and bt == Chem.BondType.DOUBLE
                and nb.GetDegree() == 1 and nb.GetFormalCharge() == 0):
            n_oxo += 1
            continue
        if bt == Chem.BondType.SINGLE:
            x_ligands.append(nb.GetIdx())
            continue
        return None  # any other bond (S=N, S#…) -> out of class
    if n_oxo not in (1, 2) or len(x_ligands) != 1:
        return None
    xp = _ligand_prefix(mol, x_ligands[0], s_idx)
    if xp is None:
        return None
    acyl = xp + ("sulfonyl" if n_oxo == 2 else "sulfinyl")
    # contractions (the -sulfinyl forms are NOT contracted::36500).
    if acyl == "hydroxysulfonyl":
        return "sulfo"
    if acyl == "aminosulfonyl":
        return "sulfamoyl"
    return acyl


def name_sulfur_oxoacid_oxy_substituent(
        mol, o_idx: int, from_idx: int) -> Optional[str]:
    """O-linked ``-O-S(oxoacid)`` as an oxy substituent prefix.

    ``o_idx`` is the ester/bridging oxygen (the fragment attach atom);
    ``from_idx`` is its parent-side neighbour. Returns ``sulfooxy`` /
    ``sulfamoyloxy`` (single compound tokens, or ``(chlorosulfonyl)oxy``
    / ``(aminosulfinyl)oxy`` / ``(methoxysulfinyl)oxy`` (an internal paren the
    caller's enclosure escalates to brackets), or ``None`` (fail closed).
    """
    o = mol.GetAtomWithIdx(o_idx)
    if (o.GetSymbol() != "O" or o.GetFormalCharge() != 0
            or o.GetTotalNumHs() != 0 or o.IsInRing() or o.GetDegree() != 2):
        return None
    s_nbrs = [n.GetIdx() for n in o.GetNeighbors()
              if n.GetIdx() != from_idx and n.GetSymbol() == "S"]
    if len(s_nbrs) != 1:
        return None
    acyl = _sulfur_oxoacid_acyl(mol, s_nbrs[0], o_idx)
    if acyl is None:
        return None
    if acyl in ("sulfo", "sulfamoyl"):
        return acyl + "oxy"          # sulfooxy / sulfamoyloxy
    return f"({acyl})oxy"            # (chlorosulfonyl)oxy, …


def name_sulfate_ester(mol, sulfur_idx: int) -> Optional[str]:
    """Functional-class name for a neutral ester of sulfuric acid.

    Sulfuric acid H2SO4 is a dibasic mononuclear noncarbon oxoacid; its
    esters are named like esters of organic acids,
    ``the Blue Book Blue Book``): the organyl group(s) are cited as
    separate words, in alphanumerical order when more than one, followed by the
    anion name of the acid ('sulfate'); a partial (mono) ester of the dibasic acid
    inserts the word 'hydrogen' (or 'dihydrogen') for a remaining acidic -OH. The
    governing PIN example is ``CH3-O-SO2-OH methyl hydrogen sulfate (PIN)``
    (:35968).

        (RO)2SO2 -> 'dialkyl sulfate' (di-ester, 0 free -OH)
        RO-SO2-OH -> 'alkyl hydrogen sulfate' (mono-ester, 1 free -OH)

    This is the SULFUR analogue of:func:`rules.phosphorus.name_phosphate_ester`
    and reuses its owner/multiplier/hydrogen-word helpers. It is distinct from a
    sulfonate ester (R-SO2-O-R', an S-C bond, already handled by 'sulfonic_ester'
    -> ``rules.esters.name_noncarbon_ester``): a sulfate ester has NO S-C bond.

    Requires a neutral, acyclic S(VI) with EXACTLY two terminal =O, at least one
    -O-C ester owner, and NO S-C bond. Fails closed (``None``) off that shape, or
    if any organyl owner is unspellable, or the name would not cover every atom;
    the top-level /OPSIN validity gate is the 0-wrong backstop on whatever
    is emitted.
    """
    from .phosphorus import (
        _HYDROGEN_MULT,
        _assemble_p_owner_text,
        _p_ester_owner_group,
    )

    if mol is None or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return None
    s = mol.GetAtomWithIdx(sulfur_idx)
    if s.GetSymbol() != "S" or s.GetFormalCharge() != 0 or s.IsInRing():
        return None

    accounted = {sulfur_idx}
    ester_oxygens: List[int] = []
    oh_count = 0
    dbl_oxo = 0
    for b in s.GetBonds():
        nb = b.GetOtherAtom(s)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE and sym == "O":
            dbl_oxo += 1
            accounted.add(nb.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return None                         # S=C / S=N / thio -> defer
        if sym == "C":
            return None                         # S-C -> sulfonate ester (other class)
        if sym != "O" or nb.GetFormalCharge() != 0:
            return None                         # S-N / charged O -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != sulfur_idx]
        if not others and nb.GetTotalNumHs() >= 1:
            oh_count += 1
            accounted.add(nb.GetIdx())
        elif len(others) == 1 and others[0].GetSymbol() == "C":
            owner_c = others[0]
            if any(ob.GetBondType() == Chem.BondType.DOUBLE
                   and ob.GetOtherAtom(owner_c).GetSymbol() == "O"
                   for ob in owner_c.GetBonds()):
                # The owner root is a carbonyl carbon -> an ACYL group
                # (R-C(=O)-O-SO2-OR'). That is a mixed carboxylic/sulfuric
                # ANHYDRIDE, not a sulfate ester -- 's
                # ester owners are alkyl/aryl groups only. Fail closed on
                # the whole molecule rather than mis-name it as a sulfate
                # ester ('acetyl methyl sulfate'); this defers to whatever
                # producer (if any) covers the mixed-anhydride class.
                return None
            ester_oxygens.append(nb.GetIdx())
            accounted.add(nb.GetIdx())
        else:
            return None                         # S-O-S bridge / O-heteroatom owner -> defer

    if dbl_oxo != 2 or not ester_oxygens:
        return None                             # not a sulfuric-acid ester (or a free acid)

    owner_tokens: List[str] = []
    for o_idx in ester_oxygens:
        got = _p_ester_owner_group(mol, o_idx, sulfur_idx)
        if got is None:
            return None
        token, frag = got
        owner_tokens.append(token)
        accounted |= frag
    owner_text = _assemble_p_owner_text(owner_tokens)

    # Every heavy atom must be covered, or this is not a whole-molecule name.
    if accounted != set(range(mol.GetNumAtoms())):
        return None

    hyd = _HYDROGEN_MULT.get(oh_count)
    if hyd is None:
        return None                             # >2 free -OH is not an ester shape
    pieces = [owner_text]
    if hyd:
        pieces.append(hyd)
    pieces.append("sulfate")
    return " ".join(pieces)


def name_sulfur_oxoacid_acyl_for_amino(
        mol, s_idx: int, frag_atoms: set) -> Optional[str]:
    """N-linked ``-NH-S(oxoacid)``: the acyl prefix for the amino wrapper.

    Reached when the S-oxoacid attaches through an amine N (the amino-branch
    caller has already split off the N as the linker and recursed into the S
    fragment, so ``s_idx`` is the fragment attach atom whose single EXTERNAL
    neighbour, outside ``frag_atoms``, is that N). Returns e.g.
    ``methoxysulfonyl`` for the caller to wrap as ``(methoxysulfonyl)amino``,
    or ``None`` (fail closed) if the external linker is not a neutral N or the S
    is out of class.
    """
    s = mol.GetAtomWithIdx(s_idx)
    ext = [n.GetIdx() for n in s.GetNeighbors() if n.GetIdx() not in frag_atoms]
    if len(ext) != 1:
        return None
    n_atom = mol.GetAtomWithIdx(ext[0])
    if n_atom.GetSymbol() != "N" or n_atom.GetFormalCharge() != 0:
        return None
    return _sulfur_oxoacid_acyl(mol, s_idx, ext[0])
