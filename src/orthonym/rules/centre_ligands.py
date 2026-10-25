"""The ligands of a mononuclear -uide or -onium centre, named by the trusted substituent-prefix namer.

The anion ``(CH3)2CH-I(-)-CH3`` is ``methyl(propan-2-yl)iodanuide``, the Blue Book): the
centre is the parent hydride ``iodane`` with the suffix 'uide', and every group on it is a detachable
prefix. The prefix of a group is what the substitutive path names it: ``assembly.substituent_enumerator.
name_substituent``, the five-tier chokepoint behind every organyl prefix of the mononuclear hydrides
(``rules.substituent_purity.organyl_prefix_name``); a ligand that holds a ring goes to the ring-substituent
chokepoint first, as it always did (``classify_substituent``'s ring branch), which spells the marks of a
nested compound prefix. ``perception.chains.classify_substituent``, which the
-uide and -onium emitters used for their ligands, names a group that is no ring by its CARBON COUNT, so
isopropyl came back 'propyl', tert-butyl 'butyl' and vinyl 'ethyl': another molecule's name.

A ligand may carry a heteroatom and a group a suffix would express: the anion, "4 Anions",
the Blue Book) or the cation ("6 Cations") of the centre outranks every neutral class a ligand holds,
so the group stays a prefix, where ``organyl_prefix_name`` fail-closes on a heteroatom because a group of
the HUB's class could be outranked by one of the organyl. A ligand with a net charge or a radical is no prefix of
the centre (a radical is senior to the anion, a cation makes the compound a zwitterion) and declines; a neutral
group with charged atoms (nitro) is a prefix.

The prefixes are assembled by the caller through ``mononuclear_hydrides._build_mixed_substituent_string``
, the Blue Book): the first cited group takes no enclosing marks, the second and every
further one is enclosed.
"""
import contextvars
from typing import List, Optional

#: True while ``assembly.substituent_enumerator._route_fragment_to_general_engine`` names a fragment capped
#: with a stand-in ``-OH``. The cap is no ligand of the molecule: its seniority-blind reading
#: ('(2-hydroxyeth-1-yn-1-yl)(phenyl)iodanium', the ion outranks the hydroxy group) is a name the route
#: turns into a prefix it cannot spell for an ion parent, and that diverted the best-effort naming of the
#: whole cation to the general engine (the same string at a lower label). A capped probe names the ligands
#: a carbon-count namer named before this module: a chain ligand with an atom that is no carbon declines.
capped_probe_ctx = contextvars.ContextVar("orthonym_capped_probe", default=False)

#: prefix of a terminal halogen (the namer of a substituent group is not asked for a single atom)
_HALOGEN_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}


def _ligand_prefix(mol, frag, attach_idx: int, ring_first: bool = True) -> Optional[str]:
    """The prefix of the ligand ``frag`` (atom indices) attached through ``attach_idx`` under the ambient tier,
    or None.

    A ligand that holds a ring is named by the ring-substituent chokepoint (``rules.ring_substituents``),
    which spells the enclosing marks of the prefixes nested in it ('2-(cycloheptyloxy)phenyl'; the five-tier
    chokepoint gives '2-cycloheptyloxyphenyl'). It does not state every stereo element of the ligand: the
    stereo is the whole name's, composed back and proven by the full-key read-back (``namer.
    _try_compose_input_stereo``), as when ``classify_substituent`` took this branch. Any other ligand, and a
    ring the chokepoint cannot name, is named by the five-tier chokepoint of the substitutive path
    (``assembly.substituent_enumerator.name_substituent``), accepted as one token with every stereo element
    stated (``rules.substituent_purity``: the acceptance of an organyl prefix). ``ring_first`` False names a
    ligand that holds a ring by the five-tier chokepoint alone: the ring-substituent chokepoint counts the
    carbons of a second ring in the ligand as a chain ('4-(octyloxy)phenyl' for 4-[(1-ethylcyclohexyl)oxy]
    phenyl), and the five-tier chokepoint spells no marks for a prefix nested in a prefix."""
    from ..assembly import substituent_enumerator
    from .ring_substituents import name_ring_system_substituent
    from .substituent_purity import _accepted_prefix_token, _is_single_prefix_token
    if any(set(ring) <= set(frag) for ring in mol.GetRingInfo().AtomRings()):
        name = name_ring_system_substituent(mol, frag, attach_idx) if ring_first else None
        if name and _is_single_prefix_token(name):
            return name
    elif capped_probe_ctx.get() and any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in frag):
        return None
    return _accepted_prefix_token(
        mol, frag, substituent_enumerator.name_substituent(mol, frag, attach_idx, allow_mancude=False))


def _is_ion_or_radical(mol, atoms) -> bool:
    """True when the ligand ``atoms`` carry a net charge or a radical electron (a neutral group with charged
    atoms, nitro or an N-oxide, is a plain prefix)."""
    return (sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in atoms) != 0
            or any(mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms))


def centre_ligand_prefixes(mol, center_idx: int, ring_first: bool = True, skip=()) -> Optional[List[str]]:
    """The prefix name of each ligand of the centre ``center_idx``, in the order of its neighbours, or None.

    None when a ligand has no name, when a ligand has a net charge or a radical, when a ligand is a ring
    through the centre, and when the heavy atoms of the centre and its ligands are not every heavy atom of
    ``mol`` (an atom a name does not cover is never shipped). An empty list is a centre with no ligand
    (``[IH2-]``). A neighbour in ``skip`` is no ligand of the name: it is a single atom the caller names
    itself (the radical centre a multiplied parent radical stands on), and counts as covered.
    """
    from .ions import _acyl_on_chalcogen_prefix, _collect_substituent_atoms, _is_acyl_on_chalcogen
    center = mol.GetAtomWithIdx(center_idx)
    names: List[str] = []
    covered = {center_idx}
    for nb in center.GetNeighbors():
        if nb.GetSymbol() == 'H':
            continue
        if nb.GetIdx() in skip:
            covered.add(nb.GetIdx())
            continue
        if nb.GetSymbol() in _HALOGEN_PREFIX and nb.GetDegree() == 1:
            names.append(_HALOGEN_PREFIX[nb.GetSymbol()])
            covered.add(nb.GetIdx())
            continue
        atoms = _collect_substituent_atoms(mol, center_idx, nb.GetIdx(), set())
        if atoms is None or _is_ion_or_radical(mol, atoms):
            return None
        frag = sorted(atoms)
        if _is_acyl_on_chalcogen(mol, nb, center_idx):
            name = _acyl_on_chalcogen_prefix(mol, nb.GetIdx(), center_idx)
        else:
            name = _ligand_prefix(mol, frag, nb.GetIdx(), ring_first)
        if not name:
            return None
        names.append(name)
        covered |= atoms
    return names if len(covered) == mol.GetNumHeavyAtoms() else None


def _breadth_contexts():
    from ..metrics.provenance import (allow_aromatic_general_ctx, best_effort_ctx, full_coverage_ctx,
                                      general_fallback_ctx)
    return (general_fallback_ctx, best_effort_ctx, allow_aromatic_general_ctx, full_coverage_ctx)


def _candidate_ligand_lists(mol, center_idx: int, skip=()):
    """The ligand prefix lists of the centre to try, lazily: the ambient tier's, then, in a call of a wider
    tier, the default tier's, each with the ring-substituent chokepoint first and then with the five-tier
    chokepoint alone (the lists that differ). No namer is right for every ligand: the best-effort namer gave
    '(methylamino)amino' for -N=N-CH3 (the double bond dropped; the default tier gives 'methyldiazenyl'), the
    default tier gave '4-decylphenyl' for 4-(adamantan-1-yl)phenyl (the best-effort ring namer gives the
    right prefix), and the ring-substituent chokepoint '4-(octyloxy)phenyl' for 4-[(1-ethylcyclohexyl)oxy]
    phenyl (the five-tier chokepoint gives the right prefix)."""
    seen = []
    for ring_first in (True, False):
        names = centre_ligand_prefixes(mol, center_idx, ring_first, skip)
        if names is not None and names not in seen:
            seen.append(names)
            yield names
    breadth = _breadth_contexts()
    if any(ctx.get() for ctx in breadth):
        for ring_first in (True, False):
            tokens = [ctx.set(False) for ctx in breadth]
            try:
                names = centre_ligand_prefixes(mol, center_idx, ring_first, skip)
            finally:
                for ctx, token in zip(breadth, tokens):
                    ctx.reset(token)
            if names is not None and names not in seen:
                seen.append(names)
                yield names


def _reads_back(mol, name: str) -> bool:
    """Whether OPSIN reads ``name`` to ``mol`` (the full InChIKey) or, for a halogen -uide name OPSIN has no
    grammar for, its method (2) spelling does (``validation.uide_twin``)."""
    from ..namer import _shipped_name_round_trip
    from ..validation.uide_twin import uide_twin_round_trip
    from rdkit import Chem
    smiles = Chem.MolToSmiles(mol)
    return _shipped_name_round_trip(name, smiles) == "verified" or uide_twin_round_trip(name, smiles) is not None


def centre_name(mol, center_idx: int, suffix: str, bare: str) -> str:
    """The name of the mononuclear centre ``center_idx``: its ligands as prefixes (the first cited group with no
    enclosing marks, the second and further ones enclosed, and ``suffix``; ``bare`` for a centre
    with no ligand; '' when a ligand has no name.

    In a call of a wider tier the name is the first of the ambient tier's and the default tier's ligand
    names that reads back (:func:`_candidate_ligand_lists`), and the first name when neither does: the
    validity gate declines it. The default tier names with its own ligand names and reads back at the gate."""
    from .mononuclear_hydrides import _build_mixed_substituent_string
    wider = any(ctx.get() for ctx in _breadth_contexts())
    first = None
    for names in _candidate_ligand_lists(mol, center_idx):
        if names is None:
            continue
        block = _build_mixed_substituent_string(names) if names else ''
        if names and not block:
            continue
        name = f"{block}{suffix}" if names else bare
        if not wider or not names or _reads_back(mol, name):
            return name
        first = first or name
    return first or ''
