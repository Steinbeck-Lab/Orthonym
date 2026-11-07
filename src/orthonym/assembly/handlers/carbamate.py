"""a phase carbamate handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:865-874 (inline branch) +
composer.py:2561-2671 (_name_carbamate body). Per internal notes, body
stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: (carbamates; functional class naming).

References:
- composer.py:865-874 (inline dispatch branch; REMOVED at this commit).
- composer.py:2561-2671 (_name_carbamate body).
- internal notes-DECOMP.md row 'carbamate' + purity proof.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from rdkit import Chem

from ..name_tree import NameTreeNode, NamingResult


def _is_carbamate(features: Any) -> bool:
    """Mirrors composer.py:865 (carbamate FG present AND principal_group is None), and also the molecule whose
    principal group ranks below the esters (:func:`_carbamate_ranks_above_principal`); the ester of carbamothioic acid
    (``assembly.carbamothioate_names``), which perception has no group for, is the same class."""
    fg = getattr(features, 'functional_groups', None) or {}
    if not fg.get('carbamate') and _thiocarbamate_unit(features) is None:
        return False
    pg = getattr(features, 'principal_group', None)
    if pg is None and not fg.get('carbamate') and not _is_neutral_species(features):
        return False
    return pg is None or _carbamate_ranks_above_principal(features, pg)


def _is_neutral_species(features: Any) -> bool:
    """False for the naming of an ion, a salt, a zwitterion or a radical. (the Blue Book-:18167) ranks
    "1 Radicals", "4 Anions", "5 Zwitterions" and "6 Cations" above "7 Acids" and "9 Esters", so the ester is not the parent of
    an ion's name. The ion routes name the neutral analogue of the ion in a nested ``name`` call and swap the suffix of
    the senior group in its name ('2-[(methoxycarbonyl)amino]ethan-1-aminium'), which needs that group, not the ester, as the
    parent: the structure of the nested call is neutral, so the molecule the caller asked about (``fragment_naming.
    outermost_name_input``) is read, and so is the molecule of ``features``."""
    from ..fragment_naming import outermost_name_input
    asked = outermost_name_input()
    if asked and not _species_is_neutral(asked):
        return False
    mol = getattr(features, 'mol', None)
    return mol is None or _species_is_neutral(Chem.MolToSmiles(mol))


@lru_cache(maxsize=4096)
def _species_is_neutral(smiles: str) -> bool:
    """``perception.ions.detect_species_type`` of ``smiles`` is 'neutral' (an unreadable string counts as neutral)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True
    from ...perception.ions import detect_species_type
    return detect_species_type(mol) == 'neutral'


def _thiocarbamate_unit(features: Any):
    """The one ester of carbamothioic or carbamodithioic acid of the molecule, as ``assembly.carbamothioate_names.
    thiocarbamate_units`` reads it; None when the molecule has none or more than one, or has an ester of carbamic acid
    as well (perception's 'carbamate' group)."""
    mol = getattr(features, 'mol', None)
    if mol is None or (getattr(features, 'functional_groups', None) or {}).get('carbamate'):
        return None
    from ..carbamothioate_names import thiocarbamate_units
    units = thiocarbamate_units(mol)
    return units[0] if len(units) == 1 else None


def _carbamate_ranks_above_principal(features: Any, pg: str) -> bool:
    """ "SENIORITY ORDER FOR CLASSES" (the Blue Book), Table 4.1 class 9 "Esters" (:18182): an acyclic
    carbamate R-O-CO-NR'2 is an ester of carbamic acid (or of hydrazinecarboxylic acid,, and an ester
    outranks the amides (class 11), hydrazides (12), nitriles (14), aldehydes (15), ketones (16), hydroxy compounds
    (17), amines (19) and every class below. (:31663): "All preferred IUPAC names for esters are named by
    functional class nomenclature", and the carbamate with a hydroxy group is the exemplar of (:30766):
    '2-hydroxypropyl (2-aminoethyl)carbamate (PIN)'. The principal group of ``features`` is the first group of
    ``rules.seniority.SENIORITY_ORDER`` that has a suffix, and a carbamate is not among them ("functional class only"),
    so a molecule whose only senior group is the carbamate has a junior ``principal_group`` -- the hydroxy group, a
    ketone, the hydrazide that perception also reads on the N-C(=O) of a carbazate -- and this handler (the carbamate
    is the parent of the name) stood aside for it. The ester of carbamothioic acid is read the same way,
    :30768; ``assembly.carbamothioate_names``).

    True when ``pg`` ranks below the esters, the molecule has no ester of a senior acid (class 7a, below), the molecule
    has the one carbamate and its C and ester chalcogen are not ring atoms (a cyclic carbamate is a heterocycle,
    ; a ring N is the ring carboxylate 'tert-butyl 4-hydroxypiperidine-1-carboxylate', which the handler
    names through ``assembly.ring_n_carboxylate``,, and the N bears no
    nitrogen but the NH2 of a hydrazinecarboxylate (``assembly.carbazate_names`` names that one)."""
    from ...rules.seniority import SENIORITY_ORDER, ester_acid_rank, seniority_index
    mol = getattr(features, 'mol', None)
    fgs = getattr(features, 'functional_groups', None) or {}
    matches = fgs.get('carbamate') or []
    thio = _thiocarbamate_unit(features)
    if mol is None or len(matches) + (thio is not None) != 1 or not _is_neutral_species(features):
        return False
    n_idx, c_idx = (matches[0][0], matches[0][1]) if matches else (thio[0], thio[1])
    ring_carboxylic = bool(matches) and mol.GetAtomWithIdx(n_idx).IsInRing()
    try:
        if seniority_index(pg, mol, getattr(features, 'principal_group_atoms', None)) <= SENIORITY_ORDER.index('ester') \
                and not (ring_carboxylic and _junior_to_ring_n_carboxylic(pg)):
            return False
    except ValueError:
        return False
    # the esters rank among themselves by their acid (rules.seniority.ESTER_ACID_ORDER): carbamic acid is a carbonic acid
    # (class 7b), below the carboxylic, sulfonic and sulfinic acids of class 7a, so the ester of one of those is the
    # parent. The groups perception reads on the carbamothioate's own carbonyl carbon (its S-ester is also a 'thioester')
    # are not such an ester. The acid on a ring nitrogen is a carboxylic acid, not a carbonic one (the Blue Book,
    # 'tetraazane-1-carboxylic acid (PIN) [not (triazan-1-yl)carbamic acid; the carboxylic acid is senior to the carbonic
    # acid derivative]'; 'pyrrolidine-1-carboxylic acid (PIN)'), first of class 7a (:18172), so only the
    # ester of another carboxylic acid can be senior to it (:func:`_junior_to_ring_n_carboxylic`).
    carbamate_rank = ester_acid_rank('carbamate')
    for name, found in fgs.items():
        rank = ester_acid_rank(name)
        if ring_carboxylic and _junior_to_ring_n_carboxylic(name):
            continue
        if rank is not None and rank < carbamate_rank and any(
                m[0] != c_idx and not _is_carbamoyl_carbon(mol, m[0]) for m in found or ()):
            return False
    c_atom = mol.GetAtomWithIdx(c_idx)
    ester_x = [nb.GetIdx() for nb in c_atom.GetNeighbors()
               if nb.GetSymbol() in ('O', 'S') and mol.GetBondBetweenAtoms(c_idx, nb.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
    if len(ester_x) != 1 or any(mol.GetAtomWithIdx(i).IsInRing() for i in (c_idx, ester_x[0])):
        return False
    from ..carbazate_names import free_hydrazine_nitrogen
    return all(nb.GetSymbol() != 'N' or nb.GetIdx() == free_hydrazine_nitrogen(mol, n_idx, c_idx)
               for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors())


#: the class-7a ester groups whose acid is a carboxylic acid with oxygen only class 7a, the Blue Book:
#: "carboxylic..., sulfonic, sulfinic..., each followed in turn by the corresponding peroxy, imidic, and hydrazonic acids.
#: Chalcogen analogues follow each of the corresponding oxygen acids"): only these can outrank the ester of a ring-nitrogen
#: carboxylic acid, by the seniority of their parent (``decomposition.ester_in_alcohol``).
_CARBOXYLIC_OXYGEN_ESTERS = frozenset({'ester', 'pseudoester'})


def _junior_to_ring_n_carboxylic(fg_name: str) -> bool:
    """True when ``fg_name`` is a class-7a ester whose acid follows a carboxylic acid in class 7a
    (:18172): the esters of the imidic and chalcogen analogues of carboxylic acids ('S-... ethanethioate') and of the
    sulfonic and sulfinic acids ('... methanesulfonate'). The ester of a ring-nitrogen carboxylic acid is senior to
    them whatever its ring ('tert-butyl 4-[(methanesulfonyl)oxy]piperidine-1-carboxylate', not '1-(tert-butoxycarbonyl)
    piperidin-4-yl methanesulfonate')."""
    from ...rules.seniority import ester_acid_rank
    return ester_acid_rank(fg_name) == 0 and fg_name not in _CARBOXYLIC_OXYGEN_ESTERS


def _is_carbamoyl_carbon(mol, idx: int) -> bool:
    """True when the carbon ``idx`` bears a nitrogen: the acyl carbon of an ester of carbamic acid or of an analogue of it,
    whose S-ester perception also reads as a 'thioester' of class 7a ('S-aryl dimethylcarbamothioate'). Carbamothioic acid is
    a carbonic acid (class 7b, "Carbonic acid", the Blue Book), and the esters of the analogues of an acid
    follow those of the oxygen acid class 7a,:18172: "Chalcogen analogues follow each of the corresponding
    oxygen acids"), so the ester of carbamic acid is the senior of the two."""
    return any(nb.GetAtomicNum() == 7 and mol.GetBondBetweenAtoms(idx, nb.GetIdx()).GetBondTypeAsDouble() == 1.0
               for nb in mol.GetAtomWithIdx(idx).GetNeighbors())


def demote_below_carbamate_parent(features: Any, name: str) -> None:
    """Record ``name``, the functional-class name of an ester of a junior acid (a phosphorus or sulfur oxoacid), as below the
    PIN when the molecule's parent is its carbamate ester (:func:`_is_carbamate`). Carbamic acid is a carbonic acid (class
    7b, "CLASS 7b. CARBON ACIDS WITH NO SUBSTITUTABLE HYDROGEN ATOMS", the Blue Book), senior to phosphonic
    and phosphinic acid (class 7c, and to phosphoric, phosphorous and sulfuric acid (class 7d,, and
    class 9 of ranks esters by their acid "Esters cited as prefixes",:31696). The
    carbamate handler runs first (priority 700); the name of the junior ester is what is left when it cannot spell an
    R group, valid and read back by OPSIN but not the PIN, so the default tier declines it."""
    if _is_carbamate(features):
        from ...metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)


def name_carbamate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B carbamate handler."""
    from ..candidate_pool import get_current_pool
    from ..carbazate_names import hydrazinecarboxylate_name
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _name_carbamate,
    )

    # R-O-C(=O)-NH-NH2 is the ester of hydrazinecarboxylic acid, which the carbamic-acid builder cannot spell
    thio = _thiocarbamate_unit(features)
    if thio is not None:
        from ..carbamothioate_names import carbamothioate_name
        carb_name = carbamothioate_name(features, thio)
    else:
        carb_name = hydrazinecarboxylate_name(features)
        if carb_name is None:
            carb_name = _name_carbamate(features)
    if not carb_name:
        # the ester of a carboxylic acid on a ring nitrogen ('ethyl piperidine-1-carboxylate',
        from ..ring_n_carboxylate import ring_n_carboxylate_ester_name
        ring_name = ring_n_carboxylate_ester_name(features.mol, style)
        if not ring_name or get_current_pool().add(ring_name, "carbamate", features) is None:
            return None
        return NamingResult(name=ring_name, tree=NameTreeNode(
            parent_stem=ring_name, class_id="carbamate", iupac_section_cite="P-65.1.2.2.2", fragment_legacy=ring_name),
            atom_to_locant_hint=None)

    carb_name = _enrich_handler_name(features, carb_name, "carbamate")

    pool = get_current_pool()
    cand = pool.add(carb_name, "carbamate", features)
    if cand is None:
        return None

    # A carbamate (and a hydrazinecarboxylate) is a retained parent with no numbered skeleton: the stereocentres of a molecule
    # that has one live inside the R and N-substituent groups, which the substituent namer cites there, so a front-of-name
    # numeric block can only resolve in a numbering the name has not "Citation of locants", the Blue Book),
    # exactly as for urea (handlers/urea.py). Without the scope the block was added in front of a name that already
    # carried the descriptors ('(2S)-benzyl [(2S)-...]carbamate'), OPSIN read it as another molecule, and the
    # name was lost to the producers that follow.
    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None,
                                           parent_scope='retained_no_locants')
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="carbamate", iupac_section_cite="P-66.4", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_carbamate", "_is_carbamate", "demote_below_carbamate_parent"]
