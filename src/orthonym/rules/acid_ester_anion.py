"""Acid-ester anion producer (IUPAC-2013.

The anion of an acid-ester of a P or S oxoacid, where the ester owner R is a
plain organyl: ``dodecyl phosphate``, ``dodecyl hydrogen phosphate``,
``diethyl phosphate``, ``dodecyl sulfate``. The protonation word is derived IN
PLACE from the surviving free ``-OH`` count; the ``[O-]`` carry the charge and
are not counted as hydrogens (— never neutralize-then-rename). Fail-closed
(``None``) off the clean single-centre ester-anion shape so a wrong molecule is
never emitted; the immediate caller (``_validate_anion_name``) only checks
suffix sanity (an ``'oic acid'`` / ``'methylidene'`` guard) — the actual
round-trip / 0-wrong backstop is the top-level /OPSIN validity gate
(``namer._final_opsin_validity_gate``).
"""
from typing import List, Optional, Set

from rdkit import Chem

# The number of PROTONATED terminal acidic oxygens -> the sulfate word .
# [1] ("hydrogen sulfate", the NEUTRAL form) is currently UNREACHABLE:
# name_sulfate_ester_anion requires anion_count >= 1, so a returned name
# always has oh_count == 0. Kept for a Slice-A follow-up (a neutral-path
# sulfate-ester producer), not dead in intent.
_SULFATE_WORD = {1: "hydrogen sulfate", 0: "sulfate"}


def name_sulfate_ester_anion(mol, sulfur_idx: int) -> Optional[str]:
    """Name the anion of a sulfate acid-ester ``R-O-S(=O)(=O)-[O-]`` -> ``{R}yl sulfate``.

    Requires: neutral non-ring S; exactly two ``=O``; exactly one ``-O-C`` ester
    owner; exactly one terminal acidic O with at least one ``[O-]``; NO S-C bond
    (that is a sulfonate — a different, already-handled class); the molecule's
    only charges are the terminal ``[O-]``; complete atom coverage. Otherwise
    ``None`` (honest fail).
    """
    from .phosphorus import _p_ester_owner_group
    if mol is None:
        return None
    s = mol.GetAtomWithIdx(sulfur_idx)
    if s.GetSymbol() != 'S' or s.GetFormalCharge() != 0 or s.IsInRing():
        return None

    accounted = {sulfur_idx}
    anion_oxygens: Set[int] = set()
    ester_oxygens: List[int] = []
    oh_count = 0
    anion_count = 0
    dbl_oxo = 0
    for b in s.GetBonds():
        nb = b.GetOtherAtom(s)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE and sym == 'O':
            dbl_oxo += 1
            accounted.add(nb.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return None                         # S=C / S=S / thio -> defer
        if sym != 'O':
            return None                         # S-C (sulfonate) / S-N -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != sulfur_idx]
        if not others and nb.GetFormalCharge() < 0:
            anion_count += 1
            anion_oxygens.add(nb.GetIdx())
            accounted.add(nb.GetIdx())
        elif not others and nb.GetTotalNumHs() >= 1 and nb.GetFormalCharge() == 0:
            oh_count += 1
            accounted.add(nb.GetIdx())
        elif len(others) == 1 and others[0].GetSymbol() == 'C' \
                and nb.GetFormalCharge() == 0:
            ester_oxygens.append(nb.GetIdx())
            accounted.add(nb.GetIdx())
        else:
            return None                         # S-O-S bridge / charged owner O -> defer

    if dbl_oxo != 2 or len(ester_oxygens) != 1 or anion_count < 1:
        return None
    # 0-wrong: ONLY the counted terminal [O-] may carry charge. A net-sum check
    # would pass a charge-separated zwitterion whose remote +/- cancel; scan
    # per-atom so any other charged centre fails closed before the owner namer.
    for a in mol.GetAtoms():
        if a.GetFormalCharge() != 0 and a.GetIdx() not in anion_oxygens:
            return None

    got = _p_ester_owner_group(mol, ester_oxygens[0], sulfur_idx)
    if got is None:
        return None
    owner, frag = got
    accounted |= frag
    if accounted != set(range(mol.GetNumAtoms())):
        return None
    word = _SULFATE_WORD.get(oh_count)
    if word is None:
        return None
    return f"{owner} {word}"


def name_oxime_o_sulfate_anion(mol, sulfur_idx: int) -> Optional[str]:
    """Name the anion of an oxime/thiohydroximate O-sulfate ester
    ``R2C=N-O-S(=O)(=O)-[O-]`` -> ``[({R2C}ylidene)amino] sulfate`` (the
    glucosinolate AGLYCONE motif, a phase; sulfate-ester-anion
    class whose ester owner is anchored at an oxime NITROGEN, not the usual
    ester-oxygen-to-carbon owner ``name_sulfate_ester_anion`` handles).

    a trace (a project rule): confirmed ``name_sulfate_ester_anion`` declines this
    shape at its OWN oxygen-classification loop (``acid_ester_anion.py``
    lines ~60-73) -- the ester oxygen's non-sulfur neighbour is an ``N``, not
    a ``C``/``S``, so it falls straight to that loop's ``else: return None``
    catch-all before ever reaching ``_p_ester_owner_group``. This sibling
    mirrors that same S-atom bond-classification shape (two ``=O``, one
    terminal ``[O-]``/``-OH``, one single-bonded ester ``-O-``) but requires
    the ester oxygen's owner to be a PLAIN oxime nitrogen (degree 2: the
    ester O plus one ``C=N`` double bond, no third N-substituent -- an
    N-substituted oxime ETHER is a different, unbuilt class and fails
    closed here).

    The owner fragment (everything reachable from the double-bonded C
    without crossing back into N) is named via the SAME ``-ylidene``
    primitive the semicarbazone/hydrazone namers use
    (``composer._double_bonded_carbon_prefix`` -- verified via
    ``free_valence_morphology`` to actually spell two free valences, never
    assumed), then wrapped ``[(...ylidene)amino]`` per the identical
    enclosure logic ``enclose_if_compound`` already applies elsewhere (BB
    verbatim precedent for the ``[(...ylidene)amino]`` construction:
    ``4-{[(4-chlorophenyl)methylidene]amino}aniline (PIN)`` under,
    ``polyfunctional.py`` ``_name_amidine_chain_side`` docstring).

    Fail-closed (``None``) off the clean single-centre shape so a wrong
    molecule is never emitted -- notably a FULL glucosinolate (the oxime
    nitrogen's owner replaced by the intact molecule, S-glycosylated with a
    thioglucoside sugar) is a completely different atom set on the S side
    (``-C(=N-O-SO3-)-S-glycosyl``) and is untouched by this function (its
    ester owner is still N here; the SUGAR sits on the unrelated S8-thiol
    side and is simply carried inside the ylidene fragment like any other
    substituent -- but the substituent namer's own sugar/ring-fragment gaps
    make that fail closed on ``_double_bonded_carbon_prefix`` returning
    ``None`` for those inputs, verified in the integration test).
    """
    from ..assembly.naming_utils import enclose_if_compound
    if mol is None:
        return None
    s = mol.GetAtomWithIdx(sulfur_idx)
    if s.GetSymbol() != 'S' or s.GetFormalCharge() != 0 or s.IsInRing():
        return None

    accounted = {sulfur_idx}
    anion_oxygens: Set[int] = set()
    ester_oxygens: List[int] = []
    oh_count = 0
    anion_count = 0
    dbl_oxo = 0
    for b in s.GetBonds():
        nb = b.GetOtherAtom(s)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE and sym == 'O':
            dbl_oxo += 1
            accounted.add(nb.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return None                         # S=C / S=S / thio -> defer
        if sym != 'O':
            return None                         # S-C (sulfonate) / S-N -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != sulfur_idx]
        if not others and nb.GetFormalCharge() < 0:
            anion_count += 1
            anion_oxygens.add(nb.GetIdx())
            accounted.add(nb.GetIdx())
        elif not others and nb.GetTotalNumHs() >= 1 and nb.GetFormalCharge() == 0:
            oh_count += 1
            accounted.add(nb.GetIdx())
        elif (len(others) == 1 and others[0].GetSymbol() == 'N'
                and others[0].GetFormalCharge() == 0):
            ester_oxygens.append(nb.GetIdx())
            accounted.add(nb.GetIdx())
        else:
            return None                         # S-O-S bridge / owner-C / charged -> defer

    if dbl_oxo != 2 or len(ester_oxygens) != 1 or anion_count < 1:
        return None
    # 0-wrong: ONLY the counted terminal [O-] may carry charge (mirrors
    # name_sulfate_ester_anion's per-atom scan -- a net-sum check would pass a
    # charge-separated zwitterion whose remote +/- cancel).
    for a in mol.GetAtoms():
        if a.GetFormalCharge() != 0 and a.GetIdx() not in anion_oxygens:
            return None

    o_idx = ester_oxygens[0]
    n_atom = next(nb for nb in mol.GetAtomWithIdx(o_idx).GetNeighbors()
                  if nb.GetIdx() != sulfur_idx)
    if n_atom.GetFormalCharge() != 0 or n_atom.IsInRing() or n_atom.GetIsotope():
        return None
    n_idx = n_atom.GetIdx()
    # The oxime N must be EXACTLY: the single bond to o_idx + one C=N double
    # bond -- no third N-substituent (an N-substituted oxime-ether shape is a
    # different, unbuilt class; decline rather than mis-name it).
    dbl_c = None
    for nb_bond in n_atom.GetBonds():
        other = nb_bond.GetOtherAtom(n_atom)
        if other.GetIdx() == o_idx:
            if nb_bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            continue
        if nb_bond.GetBondType() == Chem.BondType.DOUBLE and other.GetSymbol() == 'C':
            if dbl_c is not None:
                return None
            dbl_c = other.GetIdx()
        else:
            return None                          # a third N-substituent -> defer
    if dbl_c is None:
        return None
    accounted.add(n_idx)

    # BFS the ylidene fragment from the sp2 C, never crossing back into N
    # (mirrors _try_name_hydrazone_substitutive / _try_name_semicarbazone).
    frag = {dbl_c}
    stack = [dbl_c]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j == n_idx or j in frag:
                continue
            frag.add(j)
            stack.append(j)
    c_atom = mol.GetAtomWithIdx(dbl_c)
    ext = [nb.GetIdx() for nb in c_atom.GetNeighbors() if nb.GetIdx() not in frag]
    if ext != [n_idx]:
        return None                              # the C=N must be the ONLY exit bond

    from ..assembly.composer import _double_bonded_carbon_prefix
    ylidene = _double_bonded_carbon_prefix(mol, frag, dbl_c)
    if ylidene is None:
        return None
    accounted |= frag
    if accounted != set(range(mol.GetNumAtoms())):
        return None

    owner = enclose_if_compound(f"{enclose_if_compound(ylidene)}amino")
    word = _SULFATE_WORD.get(oh_count)
    if word is None:
        return None
    return f"{owner} {word}"


def name_acid_ester_anion(mol) -> Optional[str]:
    """Find the single qualifying P/S acid-ester centre and name its anion.

    Tries each non-ring P then each non-ring S; each builder's coverage audit
    guarantees a returned name accounts for the WHOLE molecule, so a
    poly-phosphate / multi-centre species fails closed (``None`` -> the caller
    falls through). For sulfur, ``name_sulfate_ester_anion`` (the plain
    C-anchored ester owner) is tried FIRST so its byte-identical behaviour is
    unchanged; ``name_oxime_o_sulfate_anion`` (a phase, the N-anchored
    oxime owner -- glucosinolate-type thiohydroximate/oxime O-sulfate anions)
    is only reached when that declines. ``None`` when nothing qualifies.
    """
    if mol is None:
        return None
    from .phosphorus import name_phosphate_ester_anion
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'P' and not atom.IsInRing():
            nm = name_phosphate_ester_anion(mol, atom.GetIdx())
            if nm:
                return nm
        elif sym == 'S' and not atom.IsInRing():
            nm = name_sulfate_ester_anion(mol, atom.GetIdx())
            if nm:
                return nm
            nm = name_oxime_o_sulfate_anion(mol, atom.GetIdx())
            if nm:
                return nm
    return None
