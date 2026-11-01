"""
Organometallic complex detection + hapticity perception (a phase).

This module provides RDKit-graph-based detection of metal-organometallic
complexes per IUPAC 2013 Blue Book § + IUPAC Red Book § + Salzer
1999 IUPAC Recommendations.

All functions are PREDICATE-PURE per internal notes + a phase + a phase hard invariant:
- NO mol mutation (no SanitizeMol, no UpdatePropertyCache, no
  AssignStereochemistry, no any other RDKit mutating method).
- NO module-global state read/write.
- NO exception swallowing (RDKit exceptions propagate; ValueError raised
  by compute_hapticity is caught at the handler layer per internal notes).

The integrity test ``test_side_effect_inventory_is_empty`` parametrized
over ``list(StoutClass)`` at ``tests/unit/routing/test_dispatch_table.py``
will auto-extend to ORGANOMETALLIC after Plan-02 ships and assert ````
for the new entry.

a phase (first scope-expansion phase per -07).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

from rdkit import Chem

from ..data.metal_scope import (BONDED_NAMES, CARBANION_SALTS, CONNECT_ANION, DATIVE, DISCONNECT,
                                GROUP_1, GROUP_2, ION_SALTS, has_mechanism, has_valence)
from ..data.organometallics import TERMINAL_LIGAND_PREFIXES
from .molcache import atoms_of, bonds_of  # audit 2026-09-03 (S2): per-call atom/bond tuples

# Allowlist for is_metal_element per RESEARCH
# Groups 1-17 metals/semimetals. Hydrogen EXCLUDED (never metal).
# Compiled from IUPAC periodic table (2023).
METAL_ELEMENT_SYMBOLS: FrozenSet[str] = frozenset({
    # Group 1 (alkali)
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    # Group 2 (alkaline earth)
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    # Group 13
    'B', 'Al', 'Ga', 'In', 'Tl',
    # Group 14
    'Si', 'Ge', 'Sn', 'Pb',
    # Group 15
    'As', 'Sb', 'Bi',
    # Group 16 (semimetals)
    'Te', 'Po',
    # Transition metals (Groups 3-12)
    'Sc', 'Y', 'La', 'Ac',
    'Ti', 'Zr', 'Hf', 'Rf',
    'V', 'Nb', 'Ta', 'Db',
    'Cr', 'Mo', 'W', 'Sg',
    'Mn', 'Tc', 'Re', 'Bh',
    'Fe', 'Ru', 'Os', 'Hs',
    'Co', 'Rh', 'Ir', 'Mt',
    'Ni', 'Pd', 'Pt', 'Ds',
    'Cu', 'Ag', 'Au', 'Rg',
    'Zn', 'Cd', 'Hg', 'Cn',
    # Lanthanides + Actinides (a phase.X scope-deferred but allowlist anyway)
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
})


# -04 : Group-14/13 metalloids that, when a ring member of a covalent
# heterocycle, are a Hantzsch-Widman ring parent (silolane/silole/borole/stannole),
# NOT an organometallic complex. detect_metal_complex defers these to the
# heterocycle path. Scope per the requirement: Si/Ge/Sn/Pb (Group 14) + B (Group 13).
# Deliberately EXCLUDES P (a separate phosphine-handler concern) and Al/Ga/In/Tl
# (kept under organometallic handling — out of -04 scope).
_GROUP_14_13_RING_DEFER: FrozenSet[str] = frozenset({'Si', 'Ge', 'Sn', 'Pb', 'B'})


# metal classification for mixed-metal covalent compounds:
# (1) metals of Groups 1 through 12 -> central-atom eligible;
# (2) metals of Groups 13 through 16 -> cited as a substituent group.
# When exactly ONE class-1 metal and >=1 class-2 metalloid are present in a single
# covalent component, the class-1 metal is the central atom and the class-2
# metalloid sits inside a ligand: e.g. C6H5-Hg-C6H4-Sb(C6H5)2 ->
# [4-(diphenylstibanyl)phenyl](phenyl)mercury). _CLASS_2 = Groups 13-16 metals/
# metalloids; _CLASS_1 = every other (Groups 1-12) metal in METAL_ELEMENT_SYMBOLS.
_CLASS_2_METALLOIDS: FrozenSet[str] = frozenset({
    'B', 'Al', 'Ga', 'In', 'Tl',        # Group 13
    'Si', 'Ge', 'Sn', 'Pb',             # Group 14
    'As', 'Sb', 'Bi',                   # Group 15
    'Te', 'Po',                         # Group 16
})
_CLASS_1_METALS: FrozenSet[str] = METAL_ELEMENT_SYMBOLS - _CLASS_2_METALLOIDS


def has_only_class_2_metals(mol: "Chem.Mol") -> bool:
    """True when the molecule has at least one metal atom and every one is of Groups 13 through 16
    (class (2) of, the Blue Book): Si, Ge, Sn, Pb, B, As, Sb, Bi... (:39737)
    "Organometallic compounds involving the elements of Groups 13 through 16 are named
    substitutively by prefixing the appropriate substituent names to the name of a parent hydride",
    so the seniority of classes, Table 4.1,:18158) decides the parent."""
    symbols = {a.GetSymbol() for a in atoms_of(mol) if is_metal_element(a.GetSymbol())}
    return bool(symbols) and symbols <= _CLASS_2_METALLOIDS


@dataclass(frozen=True)
class LigandGroup:
    """A contiguous set of ligand atoms coordinated to a single metal."""
    metal_atom_idx: int                                  # mol atom index
    ligand_atom_indices: Tuple[int, ...]                 # sorted tuple
    hapticity_n: int                                     # ≥ 1
    ligand_smarts_key: Optional[str] = None              # SMARTS catalog hit
    ligand_canonical_smiles: Optional[str] = None        # ligand subgraph SMILES


@dataclass(frozen=True)
class MetallacycleInfo:
    """W8-P9 Task 9.5: a metal atom that is itself a RING member.

    ``ring_atom_indices`` lists the ring atoms in TRAVERSAL order starting at
    the metal (index 0), in one of the two possible ring-walk directions
    (the assembler tries both and picks the one giving the lowest locants).
    ``exocyclic_atom_indices`` is one tuple of atom indices per exocyclic
    substituent group hanging directly off the metal (e.g. the 2 Cl atoms on
    Pt in `Cl[Pt]1(Cl)C(C)=C(C)C(C)=C1C`).
    """
    metal_atom_idx: int
    ring_atom_indices: Tuple[int, ...]
    exocyclic_atom_indices: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class MetalComplex:
    """A complete metal-organometallic complex: one or more metals + ligands."""
    metal_atom_indices: Tuple[int, ...]                  # sorted
    ligand_groups: Tuple[LigandGroup, ...]               # per-metal-per-ligand
    formal_charges: Tuple[int, ...]                      # one per metal
    is_multimetal: bool                                  # True if > 1 metal


def is_metal_element(symbol: str) -> bool:
    """Predicate: is the given element symbol a metal/semimetal?

    PURE: read-only lookup in METAL_ELEMENT_SYMBOLS frozenset.
    """
    return symbol in METAL_ELEMENT_SYMBOLS


def detect_metal_complex(mol: "Chem.Mol") -> Optional[MetalComplex]:
    """Detect any organometallic complex in the given mol.

    Plan-03 Tier-1 implementation: detects dot-separated metallocene
    topology (1 metal cation + 2 anionic Cp rings = sandwich complex).

    Plan-03 commits 03-02/03/04 extend with Tier-3 (σ-bonded
    single-component main-group), Tier-2 (dot-separated metal carbonyls),
    and Tier-4 (mixed η-bonded / half-sandwich) topology branches.

    PURE per internal notes: no mol mutation; only reads via GetAtoms /
    GetBonds / GetSymbol / GetFormalCharge / GetBondType / GetIsAromatic /
    GetOtherAtomIdx / GetBondTypeAsDouble / GetAtomWithIdx / GetMolFrags.
    NEVER: SanitizeMol, UpdatePropertyCache, AssignStereochemistry.
    """
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None

    # Collect metal atom indices + formal charges
    metal_atom_indices = []
    formal_charges = []
    for atom in atoms_of(mol):
        if is_metal_element(atom.GetSymbol()):
            metal_atom_indices.append(atom.GetIdx())
            formal_charges.append(atom.GetFormalCharge())

    if not metal_atom_indices:
        return None

    # === Tier-1 / Tier-2 topology: dot-separated multi-component ===
    # Frags structure: 1 metal frag (single atom) + N ligand frags
    # (Cp anions for Tier-1, CO ligands for Tier-2).
    frags = Chem.GetMolFrags(mol, asMols=False)

    # mixed-class single-centred complex: EXACTLY ONE class-1 (Groups 1-12)
    # central metal + >=1 class-2 (Groups 13-16) metalloid, all in ONE covalent
    # component. The class-1 metal is the sole central atom; the class-2 metalloid
    # is named as a substituent inside a ligand. This is NOT a multinuclear bridge
    # (a phase deferred). class-2/class-2 catenated hydrides (0 class-1 metals)
    # and ionic dot-separated salts (len(frags) > 1) are excluded by construction.
    _class1 = [i for i in metal_atom_indices
               if mol.GetAtomWithIdx(i).GetSymbol() in _CLASS_1_METALS]
    _class2 = [i for i in metal_atom_indices
               if mol.GetAtomWithIdx(i).GetSymbol() in _CLASS_2_METALLOIDS]
    _mixed_class_central = (
        len(frags) == 1
        and len(metal_atom_indices) >= 2
        and len(_class1) == 1
        and len(_class1) + len(_class2) == len(metal_atom_indices)
    )

    # a phase deferred: multinuclear bridged complexes (but NOT the mixed-class
    # single-centred case above, which is genuinely single-centred).
    is_multimetal = len(metal_atom_indices) > 1 and not _mixed_class_central
    if len(frags) >= 2 and not is_multimetal:
        metal_idx = metal_atom_indices[0]
        # Find metal frag and ligand frags
        metal_frag = None
        ligand_frags = []
        for frag in frags:
            if metal_idx in frag:
                metal_frag = frag
            else:
                ligand_frags.append(frag)

        # Tier-1 metallocene: metal frag is single atom + exactly 2 Cp ligand
        # frags (un-substituted or substituted via SMARTS-template match).
        if (metal_frag is not None and len(metal_frag) == 1
                and len(ligand_frags) == 2):
            cp_groups: List[LigandGroup] = []
            for lig_frag in ligand_frags:
                cp_group = _try_make_cp_group(mol, metal_idx, lig_frag)
                if cp_group is not None:
                    cp_groups.append(cp_group)
            if len(cp_groups) == 2:
                return MetalComplex(
                    metal_atom_indices=tuple(metal_atom_indices),
                    ligand_groups=tuple(cp_groups),
                    formal_charges=tuple(formal_charges),
                    is_multimetal=False,
                )

        # Tier-2 metal carbonyl: metal frag is single atom + N CO ligand frags
        if (metal_frag is not None and len(metal_frag) == 1
                and len(ligand_frags) >= 2
                and all(_is_co_ligand(mol, lf) for lf in ligand_frags)):
            co_groups = [
                LigandGroup(
                    metal_atom_idx=metal_idx,
                    ligand_atom_indices=tuple(sorted(lf)),
                    hapticity_n=1,
                    ligand_smarts_key='[C-]#[O+]',
                    ligand_canonical_smiles='[C-]#[O+]',
                )
                for lf in ligand_frags
            ]
            return MetalComplex(
                metal_atom_indices=tuple(metal_atom_indices),
                ligand_groups=tuple(co_groups),
                formal_charges=tuple(formal_charges),
                is_multimetal=False,
            )

        # Tier-4 dot-separated mixed π-ligand topology (benzene / butadiene /
        # COT / allyl / cycloheptatrienyl + optional CO ligands).
        if (metal_frag is not None and len(metal_frag) == 1
                and len(ligand_frags) >= 1):
            tier4_groups = _build_pi_ligand_groups(mol, metal_idx, ligand_frags)
            if tier4_groups is not None:
                return MetalComplex(
                    metal_atom_indices=tuple(metal_atom_indices),
                    ligand_groups=tuple(tier4_groups),
                    formal_charges=tuple(formal_charges),
                    is_multimetal=False,
                )

    # === Tier-3 topology: single-component σ-bonded main-group ===
    # 1 metal atom + N alkyl/aryl ligand atoms directly bonded (single bonds);
    # Grignard topology: 1 organic ligand + 1 halide ligand.
    if len(frags) == 1 and not is_multimetal:
        # For a mixed-class compound the class-1 metal is the sole central
        # atom and the class-2 metalloid(s) are walked THROUGH into a ligand; for
        # the ordinary single-metal case behaviour is unchanged.
        if _mixed_class_central:
            metal_idx = _class1[0]
            central_indices = (_class1[0],)
            walk_through = frozenset(
                mol.GetAtomWithIdx(i).GetSymbol() for i in _class2)
        else:
            metal_idx = metal_atom_indices[0]
            central_indices = tuple(metal_atom_indices)
            walk_through = frozenset()
        # -04 : a single ring-member Group-14/13 metalloid
        # (Si/Ge/Sn/Pb/B) in a covalent heterocycle is NOT an organometallic
        # complex — it is a Hantzsch-Widman ring parent (silolane / silole /
        # borole / stannole). Defer to the heterocycle path (CFR cascade-continue)
        # so the ring is not opened to a Si-anchored chain (`butylsilane`).
        # Scope: Si/Ge/Sn/Pb/B only (NOT P; NOT Al/Ga/In/Tl, which stay claimed).
        # Cp sandwiches are dot-separated (handled by Tier-1/4 above) and their
        # metal is NOT a ring atom, so this never mis-defers a real complex.
        # Check the CENTRAL metal(s) only — a class-2 ring metalloid inside a
        # ligand does not disqualify a genuine class-1 central complex.
        if all(
            mol.GetAtomWithIdx(i).GetSymbol() in _GROUP_14_13_RING_DEFER
            and mol.GetAtomWithIdx(i).IsInRing()
            for i in central_indices
        ):
            return None
        sigma_groups = _build_sigma_ligand_groups(mol, metal_idx, walk_through)
        if sigma_groups is not None:
            return MetalComplex(
                metal_atom_indices=central_indices,
                ligand_groups=tuple(sigma_groups),
                formal_charges=tuple(
                    mol.GetAtomWithIdx(i).GetFormalCharge() for i in central_indices),
                is_multimetal=False,
            )

    # Tier-2 (dot-separated metal carbonyls) and Tier-4 (mixed η-bonded /
    # half-sandwich) topologies land in subsequent Plan-03 commits.
    return None


_HALIDE_SMARTS_KEYS: FrozenSet[str] = frozenset({
    '[Cl-]', '[Br-]', '[F-]', '[I-]',
})


def _build_sigma_ligand_groups(mol: "Chem.Mol",
                               metal_idx: int,
                               walk_through: "FrozenSet[str]" = frozenset(),
                               ) -> Optional[Tuple[LigandGroup, ...]]:
    """Build LigandGroups for a σ-bonded single-component organometallic.

    For each direct neighbor of the metal atom, walk the connected non-metal
    subgraph (BFS over single/aromatic bonds, not crossing the metal again).
    Each component becomes a LigandGroup with hapticity_n=1.

    For halide leaf-atoms (Cl/Br/F/I directly bonded to metal), produce a
    single-atom LigandGroup with ligand_smarts_key set per
    _HALIDE_SMARTS_KEYS so the rules layer can dispatch Grignard vs alkyl.

    A metal admitted to ``BONDED_NAMES`` (``data/metal_scope.py``) also takes a terminal atom joined
    by a double bond (the =O of ``O=[Zn]``) as a ligand of its own, and a metal that carries
    hydrogens takes no ligand at all (``[ZnH2]``): the hydrogens are its ligands Compounds
    with at least one metal-carbon single bond: "The presence of hydrogen attached to a metal atom
    must always be indicated by the prefix 'hydrido'"), and an organic group joined by one double or
    triple bond from a carbon or a nitrogen atom is a ligand too ('methylidene', 'phenylimino'). Every
    other metal declines a multiple bond and a drawing without a neighbour, as before.

    Returns None if the topology doesn't match σ-bonded (e.g., aromatic
    π-bonded ligand attached to metal). Pure per internal notes.
    """
    metal_atom = mol.GetAtomWithIdx(metal_idx)
    bonded_names = has_mechanism(metal_atom.GetSymbol(), BONDED_NAMES)
    sigma_groups: List[LigandGroup] = []
    seen_atoms: set = {metal_idx}

    for neighbor_bond in metal_atom.GetBonds():
        neighbor_idx = neighbor_bond.GetOtherAtomIdx(metal_idx)
        if neighbor_idx in seen_atoms:
            continue
        neighbor_atom = mol.GetAtomWithIdx(neighbor_idx)

        # Halide leaf-atom: single-atom ligand
        if (neighbor_atom.GetDegree() == 1
                and neighbor_atom.GetSymbol() in ('Cl', 'Br', 'F', 'I')):
            halide_key = f"[{neighbor_atom.GetSymbol()}-]"
            sigma_groups.append(LigandGroup(
                metal_atom_idx=metal_idx,
                ligand_atom_indices=(neighbor_idx,),
                hapticity_n=1,
                ligand_smarts_key=halide_key,
                ligand_canonical_smiles=halide_key,
            ))
            seen_atoms.add(neighbor_idx)
            continue

        # A terminal atom of a metal admitted to BONDED_NAMES, joined by a double bond, that the
        # table of terminal ligands names (=O, =S): a ligand of its own. Any other doubly bonded
        # atom (=CH2, =NH) stays declined, as for every other metal.
        if (bonded_names and neighbor_bond.GetBondType() == Chem.BondType.DOUBLE
                and neighbor_atom.GetDegree() == 1 and neighbor_atom.GetFormalCharge() == 0
                and neighbor_atom.GetIsotope() == 0
                and (neighbor_atom.GetSymbol(), 2, neighbor_atom.GetTotalNumHs())
                in TERMINAL_LIGAND_PREFIXES):
            sigma_groups.append(LigandGroup(
                metal_atom_idx=metal_idx,
                ligand_atom_indices=(neighbor_idx,),
                hapticity_n=1,
                ligand_smarts_key=None,
                ligand_canonical_smiles=None,
            ))
            seen_atoms.add(neighbor_idx)
            continue

        # Skip if connecting via a non-single bond (aromatic / double / triple)
        # — those are π-bonded, Tier-4 territory. Aryl σ-bonds to metal are
        # single bonds; aromatic-ring atoms inside the ring are aromatic but
        # the M-C bond itself is single. A metal admitted to BONDED_NAMES also takes an organic group
        # joined by one double or triple bond from a carbon or a nitrogen atom (=CH2, =CH-C6H5, #C-CH3,
        # =N-C6H5): its ligand name is the 'ylidene', 'ylidyne' or 'imino' prefix of
        # Definitions of terms, "names for substituent groups as defined in Section can also be used
        # as ligand names"; ``rules.organometallics._multiply_bonded_ligand_name``).
        multiple = (bonded_names and neighbor_bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
                    and neighbor_atom.GetSymbol() in ('C', 'N'))
        if neighbor_bond.GetBondTypeAsDouble() > 1.5 and not multiple:
            return None

        # Walk the organic ligand subgraph via BFS, not crossing the central
        # metal; class-2 metalloids in ``walk_through`` are traversed (they are
        # part of this ligand, e.g. the Sb of a diphenylstibanyl-phenyl ligand).
        ligand_atoms = _collect_organic_component(
            mol, neighbor_idx, metal_idx, walk_through)
        # Refuse to collect atoms already in another ligand (e.g., a bridging
        # carbon would belong to two metals — a phase territory)
        if any(a in seen_atoms for a in ligand_atoms):
            return None
        seen_atoms.update(ligand_atoms)

        sigma_groups.append(LigandGroup(
            metal_atom_idx=metal_idx,
            ligand_atom_indices=tuple(sorted(ligand_atoms)),
            hapticity_n=1,
            ligand_smarts_key=None,  # σ-bonded; no SMARTS catalog hit
            ligand_canonical_smiles=None,
        ))

    if not sigma_groups:
        return () if bonded_names and metal_atom.GetTotalNumHs() > 0 else None
    return tuple(sigma_groups)


def _collect_organic_component(mol: "Chem.Mol", start_idx: int,
                                metal_idx: int,
                                walk_through: "FrozenSet[str]" = frozenset(),
                                ) -> List[int]:
    """BFS over non-metal atoms reachable from start_idx, not crossing the central
    metal. Metals whose symbol is in ``walk_through`` (class-2 metalloids that are
    ligand-internal substituents, e.g. an Sb in a diphenylstibanyl group) ARE
    traversed and included; every other metal atom stops the walk (a separate
    coordination centre)."""
    visited = {start_idx}
    queue = [start_idx]
    while queue:
        current = queue.pop()
        atom = mol.GetAtomWithIdx(current)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(current)
            if other_idx == metal_idx:
                continue
            if other_idx in visited:
                continue
            other_atom = mol.GetAtomWithIdx(other_idx)
            if (other_atom.GetSymbol() not in walk_through
                    and is_metal_element(other_atom.GetSymbol())):
                continue
            visited.add(other_idx)
            queue.append(other_idx)
    return sorted(visited)


def _try_make_cp_group(mol: "Chem.Mol", metal_idx: int,
                       atom_indices: Tuple[int, ...]
                       ) -> Optional[LigandGroup]:
    """Try to recognise an atom-frag as a Cp ligand (with optional substituents).

    Recognises:
    - Bare Cp anion ``c1cc[cH-]c1`` (5 atoms)
    - Substituted Cp anion via SMARTS-template lookup in LIGAND_ETA_DEFAULTS
      (e.g. pentamethyl-Cp ``Cc1c(C)c(C)[c-](C)c1C``)

    Returns None if the frag doesn't match any known Cp variant.
    """
    from ..data.organometallics import LIGAND_ETA_DEFAULTS

    if _is_cp_ligand(mol, atom_indices):
        return LigandGroup(
            metal_atom_idx=metal_idx,
            ligand_atom_indices=tuple(sorted(atom_indices)),
            hapticity_n=5,
            ligand_smarts_key='c1cc[cH-]c1',
            ligand_canonical_smiles='c1cc[cH-]c1',
        )

    # Substituted Cp: compute frag canonical SMILES and look up
    submol = Chem.RWMol()
    atom_map: dict = {}
    for idx in atom_indices:
        new_idx = submol.AddAtom(mol.GetAtomWithIdx(idx))
        atom_map[idx] = new_idx
    for idx in atom_indices:
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other in atom_indices and other > idx:
                submol.AddBond(atom_map[idx], atom_map[other], bond.GetBondType())
    try:
        ligand_smi = Chem.MolToSmiles(submol)
    except Exception:
        return None

    if ligand_smi in LIGAND_ETA_DEFAULTS:
        hapticity, name = LIGAND_ETA_DEFAULTS[ligand_smi]
        # Accept only Cp-class ligands (hapticity 5; named with 'cyclopentadienyl' suffix)
        if hapticity == 5 and 'cyclopentadienyl' in name:
            return LigandGroup(
                metal_atom_idx=metal_idx,
                ligand_atom_indices=tuple(sorted(atom_indices)),
                hapticity_n=5,
                ligand_smarts_key=ligand_smi,
                ligand_canonical_smiles=ligand_smi,
            )
    return None


def _is_cp_ligand(mol: "Chem.Mol", atom_indices: Tuple[int, ...]) -> bool:
    """Check if a frag's atom indices form a Cp anion ring.

    Cp anion = 5 aromatic carbon atoms in a single ring with sum of
    formal charges = -1. Pure read-only on mol.
    """
    if len(atom_indices) != 5:
        return False
    charge_sum = 0
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return False
        if not atom.GetIsAromatic():
            return False
        charge_sum += atom.GetFormalCharge()
    return charge_sum == -1


def _build_pi_ligand_groups(mol: "Chem.Mol", metal_idx: int,
                             ligand_frags: List[Tuple[int, ...]]
                             ) -> Optional[List[LigandGroup]]:
    """Build LigandGroups for Tier-4 mixed π-ligand topology.

    Each ligand frag is a contiguous π-system bound to the metal. Use
    LIGAND_ETA_DEFAULTS for SMARTS-template lookup; on miss return None
    (handler cascades to SALT@100). Also accept CO frags (will already
    have been caught by Tier-2 if all-CO; this branch handles mixed
    CO + π topology).
    """
    from ..data.organometallics import LIGAND_ETA_DEFAULTS

    groups: List[LigandGroup] = []
    for frag in ligand_frags:
        # CO ligand sub-case
        if _is_co_ligand(mol, frag):
            groups.append(LigandGroup(
                metal_atom_idx=metal_idx,
                ligand_atom_indices=tuple(sorted(frag)),
                hapticity_n=1,
                ligand_smarts_key='[C-]#[O+]',
                ligand_canonical_smiles='[C-]#[O+]',
            ))
            continue

        # Cp anion (5 aromatic C, charge sum -1)
        if _is_cp_ligand(mol, frag):
            groups.append(LigandGroup(
                metal_atom_idx=metal_idx,
                ligand_atom_indices=tuple(sorted(frag)),
                hapticity_n=5,
                ligand_smarts_key='c1cc[cH-]c1',
                ligand_canonical_smiles='c1cc[cH-]c1',
            ))
            continue

        # General π-ligand: compute canonical SMILES of the sub-frag and
        # look up in LIGAND_ETA_DEFAULTS
        submol = Chem.RWMol()
        atom_map: dict = {}
        for idx in frag:
            new_idx = submol.AddAtom(mol.GetAtomWithIdx(idx))
            atom_map[idx] = new_idx
        for idx in frag:
            for bond in mol.GetAtomWithIdx(idx).GetBonds():
                other = bond.GetOtherAtomIdx(idx)
                if other in frag and other > idx:
                    submol.AddBond(atom_map[idx], atom_map[other], bond.GetBondType())
        try:
            ligand_smi = Chem.MolToSmiles(submol)
        except Exception:
            return None

        if ligand_smi in LIGAND_ETA_DEFAULTS:
            hapticity, _name = LIGAND_ETA_DEFAULTS[ligand_smi]
            groups.append(LigandGroup(
                metal_atom_idx=metal_idx,
                ligand_atom_indices=tuple(sorted(frag)),
                hapticity_n=hapticity,
                ligand_smarts_key=ligand_smi,
                ligand_canonical_smiles=ligand_smi,
            ))
            continue

        # No match — cascade to SALT@100
        return None

    if not groups:
        return None
    return groups


def _is_co_ligand(mol: "Chem.Mol", atom_indices: Tuple[int, ...]) -> bool:
    """Check if a frag's atom indices form a carbonyl (CO) ligand.

    CO ligand canonical SMILES: ``[C-]#[O+]`` — 1 C atom (charge -1) +
    1 O atom (charge +1) joined by a triple bond. Pure read-only on mol.
    """
    if len(atom_indices) != 2:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in atom_indices]
    symbols = sorted(a.GetSymbol() for a in atoms)
    if symbols != ['C', 'O']:
        return False
    c_atom = next(a for a in atoms if a.GetSymbol() == 'C')
    o_atom = next(a for a in atoms if a.GetSymbol() == 'O')
    if c_atom.GetFormalCharge() != -1:
        return False
    if o_atom.GetFormalCharge() != 1:
        return False
    # Verify the bond is triple (CO ligand has C≡O)
    bond = mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1])
    if bond is None:
        return False
    return bond.GetBondType() == Chem.BondType.TRIPLE


def compute_hapticity(mol: "Chem.Mol", metal_atom_idx: int,
                      ligand_atom_indices: Tuple[int, ...]) -> int:
    """Compute the hapticity of a ligand group bound to a metal.

    Plan-03 implementation per internal notes hybrid:
    1. SMARTS-template fast path: consult LIGAND_ETA_DEFAULTS from
       data/organometallics.py; on hit, return catalog value.
    2. Graph-walk fallback (added in 03-04 Tier-4 commit): BFS over
       ligand_atom_indices counting contiguous π-system atoms.
    3. Edge case σ-bonded (single atom or no π-system): return 1.

    PURE per internal notes.
    """
    # Lazy import to avoid circular dependency
    from ..data.organometallics import LIGAND_ETA_DEFAULTS

    if len(ligand_atom_indices) == 0:
        raise ValueError(f"empty ligand_atom_indices for metal {metal_atom_idx}")

    # Single-atom ligand: σ-bonded hapticity 1
    if len(ligand_atom_indices) == 1:
        return 1

    # Build the ligand subgraph as a separate mol (PURE per: submol only)
    submol = Chem.RWMol()
    atom_map: dict = {}
    for idx in ligand_atom_indices:
        # AddAtom returns the new atom's index in submol; copy from mol's atom.
        new_idx = submol.AddAtom(mol.GetAtomWithIdx(idx))
        atom_map[idx] = new_idx
    for idx in ligand_atom_indices:
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other in ligand_atom_indices and other > idx:
                submol.AddBond(atom_map[idx], atom_map[other], bond.GetBondType())

    # SMARTS-template fast path lookup
    try:
        ligand_smi = Chem.MolToSmiles(submol)
    except Exception:
        ligand_smi = None

    if ligand_smi and ligand_smi in LIGAND_ETA_DEFAULTS:
        return LIGAND_ETA_DEFAULTS[ligand_smi][0]

    # Graph-walk fallback (Tier-4 implementation lands in commit 03-04)
    # For Tier-1 (Cp via SMARTS fast path), this never executes.
    return 1


# W8-P9 Task 9.2: the TRUE-METAL subset of METAL_ELEMENT_SYMBOLS used by
# has_covalent_metal_carbon_bond — Groups 1, 2, and 3-12 (+ lanthanides/
# actinides), per BB 's own three-way split: "(1) elements of Groups 1
# and 2; (2) elements of Groups 3-12 (the transition metals); (3) elements
# of Groups 13-16". Deliberately EXCLUDES the Group 13-16 metalloids (B, Al,
# Ga, In, Tl, Si, Ge, Sn, Pb, As, Sb, Bi, Te, Po): those ALWAYS have a
# legitimate alternate SUBSTITUTIVE nomenclature system that correctly
# names the WHOLE molecule via a different CFR class —
# MONONUCLEAR_HYDRIDE (trimethylarsane/trimethylgallane/trimethylindigane/
# trimethylthallane), the -04 Hantzsch-Widman ring defer (silole/borole/
# stannole), FREE_HOMONUCLEAR_G14_HYDRIDE, DINUCLEAR_HYDRIDE, etc. — so a
# "class_id != organometallic" outcome for one of THOSE elements is not an
# atom-drop leak; it is the correct alternate path. Restricting the veto's
# metal scope to true metals (which have NO alternate substitutive system in
# this codebase) makes the veto SAFE: a true-metal-carbon bond can only be
# legitimately named via additive/coordination nomenclature, so any
# other outcome for it genuinely is a structure-loss leak.
_TRUE_METAL_SYMBOLS_FOR_VETO: FrozenSet[str] = METAL_ELEMENT_SYMBOLS - frozenset({
    'B', 'Al', 'Ga', 'In', 'Tl',       # Group 13
    'Si', 'Ge', 'Sn', 'Pb',            # Group 14
    'As', 'Sb', 'Bi',                  # Group 15
    'Te', 'Po',                        # Group 16
})


def assembly_has_out_of_scope_metal(mol: "Chem.Mol") -> bool:
    """True if a multi-fragment ionic assembly contains a true metal that the scope table
    (``data/metal_scope.py``) does not admit, so the best-effort ion/adduct widening must NOT
    fire (0-wrong: never render a coordination complex / organometallic as a salt or adduct).

    OUT <=> (i) a true metal (``_TRUE_METAL_SYMBOLS_FOR_VETO``) that is not admitted to the
    ``ION_SALTS`` mechanism, OR (ii) a carbanion or a hydride is present and a true metal is
    not admitted to the ``CARBANION_SALTS`` mechanism: INTRODUCTION, "Organometallic
    compounds are compounds having at least one bond between one metal atom and one carbon
    atom", so such an assembly is organometallic for that metal.
    Metalloids (Si/Ge/Sn/B/As/... -- excluded from ``_TRUE_METAL_SYMBOLS_FOR_VETO``)
    are IN scope and never trip this.

    A salt or an adduct of a Group 1 or Group 2 cation with organic ions is IN scope
    binary salt / adduct); those elements are the ones the table admits to
    ``ION_SALTS``.
    """
    true_metals = {a.GetSymbol() for a in mol.GetAtoms()
                   if a.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO}
    if not true_metals:
        return False
    if any(not has_mechanism(symbol, ION_SALTS) for symbol in true_metals):
        return True
    if any(not has_mechanism(symbol, CARBANION_SALTS) for symbol in true_metals):
        return any((a.GetSymbol() == 'C' and a.GetFormalCharge() < 0)
                   or (a.GetSymbol() == 'H' and a.GetFormalCharge() < 0)
                   for a in mol.GetAtoms())
    return False


#: the true metals of Groups 3-12 (the transition elements including Group 3, the lanthanides and the
#: actinides): the true metals without Groups 1 and 2. ORGANOMETALLIC COMPOUNDS INVOLVING THE
#: ELEMENTS IN GROUPS 3 THROUGH 12 puts them together.
GROUP_3_12_METAL_SYMBOLS: FrozenSet[str] = (
    _TRUE_METAL_SYMBOLS_FOR_VETO - frozenset(GROUP_1) - frozenset(GROUP_2))


def _is_cyanide_carbon(atom: "Chem.Atom") -> bool:
    """True for the carbon of a cyanide anion: one heavy neighbour, a nitrogen joined by a triple
    bond, no hydrogen."""
    if atom.GetDegree() != 1 or atom.GetTotalNumHs() != 0:
        return False
    bond = atom.GetBonds()[0]
    other = bond.GetOtherAtom(atom)
    return (other.GetSymbol() == 'N' and other.GetFormalCharge() == 0
            and bond.GetBondType() == Chem.BondType.TRIPLE)


#: the enolate of a beta-dicarbonyl compound, the anion of 'acetylacetone': O(-)-C=C-C=O
_BETA_DIKETONATE = Chem.MolFromSmarts("[O-]-[#6]=[#6]-[#6]=O")


def holds_beta_diketonate_salt_of_group_3_12_metal(mol: "Chem.Mol") -> bool:
    """True iff ``mol`` holds a true metal of Groups 3-12 with no bond to another atom and an anion of the beta-diketonate
    kind ``O(-)-C=C-C=O`` (the acetylacetonate ligand of a chelate drawn ionic).

     (the Blue Book) puts the compounds of the Group 3-12 metals with their ligands into coordination
    nomenclature, and (:39735) notes no PIN for them; an enolate of this kind is the chelating ligand of the
    compound, drawn here as the salt it is written as. A salt of a metal of Group 1 or 2 keeps the label of a salt of a
    hydroxy compound,:28178). Structure only; the name-dependent exclusions are the caller's. PURE."""
    if mol is None:
        return False
    metals = [a for a in mol.GetAtoms() if a.GetSymbol() in GROUP_3_12_METAL_SYMBOLS]
    if not metals or any(a.GetDegree() for a in metals):
        return False
    return mol.HasSubstructMatch(_BETA_DIKETONATE)


def holds_carbanion_salt_of_group_3_12_metal(mol: "Chem.Mol") -> bool:
    """True iff ``mol`` is, structurally, a salt of a carbanion with a Group 3-12 metal cation: it
    holds a true metal of Groups 3-12, has no bond from a true metal to carbon, and a fragment of
    negative net charge holds a carbon atom with formal charge -1 that has no positively charged
    neighbour and is not the carbon of a cyanide anion.

     INTRODUCTION (the Blue Book): "Organometallic compounds are compounds having at
    least one bond between one metal atom and one carbon atom"; a salt of a carbanion has no such
    bond (it is named by composition,,:40173: "it is a compositional name analogous to
    sodium chloride"), and the status of its name is that of the organometallic compounds of the
    metal: (:39735) "neither preferred IUPAC names or preselected names... for
    organometallic compounds involving the transition elements (including the Group 3 elements)
    and Groups 1 and 2 elements, except for 'ocene' compounds, are noted". A carbon with a
    positively charged neighbour is an ylide or an isocyanide, and the carbon of a cyanide anion is
    the anion of hydrogen cyanide, an acid: neither is a carbanion here.

    Structure only: the name-dependent exclusions (the retained and the additive names) are the
    predicate of the namer, which calls this. PURE: no mol mutation."""
    if mol is None:
        return False
    if not any(a.GetSymbol() in GROUP_3_12_METAL_SYMBOLS for a in mol.GetAtoms()):
        return False
    if has_covalent_metal_carbon_bond(mol):
        return False
    for fragment in Chem.GetMolFrags(mol):
        if sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in fragment) >= 0:
            continue
        for i in fragment:
            atom = mol.GetAtomWithIdx(i)
            if (atom.GetSymbol() == 'C' and atom.GetFormalCharge() == -1
                    and not any(n.GetFormalCharge() > 0 for n in atom.GetNeighbors())
                    and not _is_cyanide_carbon(atom)):
                return True
    return False


def single_bond_metal_heteroatom_sites(mol: "Chem.Mol") -> Optional[List[Tuple[int, int]]]:
    """The bonds of ``mol`` that a charge-conserving heterolytic disconnection may break, as
    ``(metal atom index, other atom index)`` pairs; ``None`` when ``mol`` is not a drawing that
    ``DISCONNECT`` (``data/metal_scope.py``) covers.

    The sites are the single bonds from a true metal (``_TRUE_METAL_SYMBOLS_FOR_VETO``) to an
    atom that is neither carbon nor a metal or metalloid. ``None`` (the drawing is left as it is
    written) when

    * there is no such bond;
    * a true metal is bonded to carbon INTRODUCTION, "Organometallic compounds are
      compounds having at least one bond between one metal atom and one carbon atom": the key of
      such a drawing is not the key of its ionic drawing), to another metal or a metalloid;
    * a bond from a true metal is not a single bond (a double bond, a dative bond);
    * a bonded metal is not admitted to ``DISCONNECT``, or the charge it would carry after the
      disconnection (its formal charge plus the number of its bonds broken) is not one of its
      ``valences`` in the scope table: ``Cl[Zn](Cl)N`` would give ``[Zn+3]`` with the key of
      ``Cl[Zn](Cl)N`` unchanged, and a standard InChIKey does not bound the charge of a metal;
    * a broken bond carries a bond direction, or an atom at either end carries a chiral tag, or a
      double bond at the other end carries a stereo label: breaking the bond would change the
      configuration the drawing states.

    A metal cation that has no bond (the counter-ion ``[Na+]`` of ``[Na+].Cl[Zn-2](Cl)(Cl)Cl``)
    is not a site and does not matter. PURE: read-only inspection of ``mol``.
    """
    if mol is None:
        return None
    sites: List[Tuple[int, int]] = []
    broken: dict = {}
    for b in bonds_of(mol):
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        m1 = a1.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO
        m2 = a2.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO
        if not (m1 or m2):
            continue
        if m1 and m2:
            return None
        metal, other = (a1, a2) if m1 else (a2, a1)
        symbol = other.GetSymbol()
        if symbol == 'C' or is_metal_element(symbol):
            return None
        if b.GetBondType() != Chem.BondType.SINGLE or b.GetBondDir() != Chem.BondDir.NONE:
            return None
        if (metal.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                or other.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
            return None
        if any(nb.GetStereo() != Chem.BondStereo.STEREONONE for nb in other.GetBonds()):
            return None
        sites.append((metal.GetIdx(), other.GetIdx()))
        broken[metal.GetIdx()] = broken.get(metal.GetIdx(), 0) + 1
    if not sites:
        return None
    for metal_idx, count in broken.items():
        atom = mol.GetAtomWithIdx(metal_idx)
        symbol = atom.GetSymbol()
        if not has_mechanism(symbol, DISCONNECT) or not has_valence(symbol, atom.GetFormalCharge() + count):
            return None
    return sites


#: the bond types RDKit reads for the ``->`` and ``<-`` of a SMILES; the begin atom of the bond is
#: the donor and the end atom the acceptor
_DATIVE_BOND_TYPES = (Chem.BondType.DATIVE, Chem.BondType.DATIVEONE, Chem.BondType.DATIVEL,
                      Chem.BondType.DATIVER)


def dative_bond_split_sites(mol: "Chem.Mol") -> Optional[List[Tuple[int, int]]]:
    """The dative bonds of ``mol`` that a split removes, as ``(metal atom index, donor atom index)``
    pairs; ``None`` when ``mol`` is not a drawing that ``DATIVE`` (``data/metal_scope.py``) covers.

    A drawing with dative bonds has no standard InChIKey in RDKit, so it has no key to read a name
    back against. The split drawing is the same drawing without those bonds: the donor keeps its
    lone pair, the acceptor its charge, every atom its hydrogens, and no charge moves.
    Structures for Lewis 'adducts' draws the 'dative' bond as structure (I) of the three drawings
    of an adduct. The sites are every dative bond of ``mol``; ``None`` (the drawing is left as it
    is written) when

    * there is no dative bond;
    * a dative bond does not end at a true metal (``_TRUE_METAL_SYMBOLS_FOR_VETO``): the split
      would leave a dative bond in the drawing, and the drawing would still have no key; or its
      donor is a metal or a metalloid;
    * an acceptor metal is not admitted to ``DATIVE``;
    * a bond carries a bond direction, an atom at either end of a dative bond carries a chiral tag,
      or a double bond at the donor carries a stereo label: the split would change the
      configuration the drawing states, and the key of the split drawing cannot tell.

    PURE: read-only inspection of ``mol``.
    """
    if mol is None:
        return None
    sites: List[Tuple[int, int]] = []
    for b in bonds_of(mol):
        if b.GetBondType() not in _DATIVE_BOND_TYPES:
            continue
        donor, acceptor = b.GetBeginAtom(), b.GetEndAtom()
        if (acceptor.GetSymbol() not in _TRUE_METAL_SYMBOLS_FOR_VETO
                or donor.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO
                or is_metal_element(donor.GetSymbol())
                or not has_mechanism(acceptor.GetSymbol(), DATIVE)):
            return None
        if b.GetBondDir() != Chem.BondDir.NONE:
            return None
        if (donor.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                or acceptor.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
            return None
        if any(nb.GetStereo() != Chem.BondStereo.STEREONONE for nb in donor.GetBonds()):
            return None
        sites.append((acceptor.GetIdx(), donor.GetIdx()))
    return sites or None


def connect_anion_to_organometal_cation_sites(mol: "Chem.Mol") -> Optional[List[Tuple[int, int]]]:
    """The bonds that join a free anion to the organometallic cation it is drawn beside, as
    ``(metal atom index, donor atom index)`` pairs; ``None`` when ``mol`` is not a drawing that
    ``CONNECT_ANION`` (``data/metal_scope.py``) covers.

    The drawing is ``[Cl-].[Zn+]c1ccccc1``: one cation of a true metal that carries a bond to
    carbon INTRODUCTION, "Organometallic compounds are compounds having at least one bond
    between one metal atom and one carbon atom") and the free anion that neutralises it. Each site
    is the bond from the metal to the donor atom of such an anion. ``None`` (the drawing is left as
    it is written) unless

    * exactly one true metal atom is positively charged, and no other true metal atom is in the
      molecule but a metal that has no bond at all;
    * that metal has only single bonds without a bond direction, no chiral tag, a bond to carbon,
      no bond to another metal or a metalloid, is admitted to ``CONNECT_ANION`` and its formal
      charge plus its number of bonds is one of its ``valences`` in the table (the charge the
      metal would carry if every bond were broken; the cation the anion neutralises is the metal's
      formal charge);
    * every fragment of negative net charge but the cation's is such a free anion, and they are as
      many as the metal's positive charge: a fragment of its own with net charge -1 and one
      negatively charged atom (the donor, neither carbon nor a metal or a metalloid, with no chiral
      tag and no bond that carries a stereo label), so which atom takes the bond is not a choice
      and no anion has to be picked among several. The fragment of the cation has the net charge
      of the cation.

    A carbanion is no donor: a bond from the metal to its carbon would be a second metal-carbon
    bond, and the key of that drawing is not the key of this one. PURE: read-only inspection of
    ``mol``.
    """
    if mol is None:
        return None
    # the true metals, and a Group 14 metal that the table admits to CONNECT_ANION (tin, lead: the bonded drawing
    # is a substituted stannane or plumbane,
    metals = [a for a in atoms_of(mol) if a.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO
              or (a.GetSymbol() in ('Sn', 'Pb') and has_mechanism(a.GetSymbol(), CONNECT_ANION))]
    cations = [a for a in metals if a.GetFormalCharge() > 0]
    if len(cations) != 1 or any(a.GetDegree() for a in metals if a.GetIdx() != cations[0].GetIdx()):
        return None
    cation = cations[0]
    symbol, charge = cation.GetSymbol(), cation.GetFormalCharge()
    if (not has_mechanism(symbol, CONNECT_ANION)
            or not has_valence(symbol, charge + cation.GetDegree())
            or cation.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
        return None
    bonded_to_carbon = False
    for bond in cation.GetBonds():
        other = bond.GetOtherAtom(cation)
        if (bond.GetBondType() != Chem.BondType.SINGLE or bond.GetBondDir() != Chem.BondDir.NONE
                or is_metal_element(other.GetSymbol())):
            return None
        bonded_to_carbon = bonded_to_carbon or other.GetSymbol() == 'C'
    if not bonded_to_carbon:
        return None
    sites: List[Tuple[int, int]] = []
    for fragment in Chem.GetMolFrags(mol):
        atoms = [mol.GetAtomWithIdx(i) for i in fragment]
        net = sum(a.GetFormalCharge() for a in atoms)
        if cation.GetIdx() in fragment:
            if net != charge:                       # the cation is already balanced inside its fragment
                return None
            continue
        if net >= 0:
            continue
        negative = [a for a in atoms if a.GetFormalCharge() < 0]
        donor = negative[0]
        if (net != -1 or len(negative) != 1 or donor.GetSymbol() == 'C'
                or is_metal_element(donor.GetSymbol())
                or any(a.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO for a in atoms)
                or donor.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in donor.GetBonds())):
            return None                             # an anion that is no free anion: the pairing is a choice
        if not _charge_stays_on_the_donor(mol, fragment, donor.GetIdx()):
            return None                             # an ambident anion: the drawing does not fix the donor
        sites.append((cation.GetIdx(), donor.GetIdx()))
    return sites if len(sites) == charge else None


#: the most resonance structures read of one anion (``_charge_stays_on_the_donor``); a larger anion declines
_MAX_ANION_RESONANCE_STRUCTURES = 64


def _charge_stays_on_the_donor(mol: "Chem.Mol", fragment, donor_idx: int) -> bool:
    """True when every resonance structure of the anion ``fragment`` (atom indices of ``mol``) that puts its negative
    charge on an atom other than carbon is the same structure (one canonical SMILES). The pyridin-2-olate anion
    drawn ``[O-]c1ccccn1`` and ``O=c1cccc[n-]1`` is one anion with one standard InChIKey, and a bond to its oxygen
    or to its nitrogen gives two constitutional isomers ('...[(pyridin-2-yl)oxy]stannane', '1-(...stannyl)pyridin-
    2(1H)-one') with one key too: the drawn position of the charge is no structure, so the read-back cannot tell the
    bonded drawing it would choose. The two structures of a carboxylate are one SMILES, so acetate stays connectable;
    a structure with the charge on a carbon is no other donor (a carbanion is no donor, see the caller). False when
    RDKit cannot enumerate the structures within ``_MAX_ANION_RESONANCE_STRUCTURES`` (fail closed)."""
    try:
        mapping: List[Tuple[int, ...]] = []
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mapping)
        frag = frags[next(k for k, m in enumerate(mapping) if donor_idx in m)]
        supplier = Chem.ResonanceMolSupplier(frag, Chem.ALLOW_CHARGE_SEPARATION,
                                             _MAX_ANION_RESONANCE_STRUCTURES)
        count = len(supplier)
        if count >= _MAX_ANION_RESONANCE_STRUCTURES:
            return False
        forms = set()
        for k in range(count):
            structure = supplier[k]
            if structure is None:
                return False
            if any(a.GetFormalCharge() < 0 and a.GetSymbol() != 'C' for a in structure.GetAtoms()):
                copy = Chem.Mol(structure)
                Chem.SanitizeMol(copy)
                forms.add(Chem.MolToSmiles(copy))
        return len(forms) <= 1
    except Exception:  # noqa: BLE001 -- any RDKit failure declines the connection
        return False


def atom_and_charge_ledger_ok(before: "Chem.Mol", after: "Chem.Mol",
                              changed_bonds: List[Tuple[int, int]], moved_charge: int = 1) -> bool:
    """True when ``after`` is ``before`` with the bonds ``changed_bonds`` broken and nothing else
    changed but the charges those bonds carry. A count of atoms and charges that uses no InChI
    and shares no code with the rewrite it checks: a rewrite that keeps the standard InChIKey
    (which is blind to metal-ligand bonds and to which atom of an ambident anion carries the
    charge) and moves a charge to the wrong donor atom, or drops a hydrogen, fails here.

    ``before`` and ``after`` are sanitized molecules with the same atom order. ``changed_bonds``
    lists the broken bonds as ``(metal atom index, other atom index)``: the metal gains
    ``moved_charge`` and the other atom loses it for each (+1 and -1 for the heterolytic
    disconnection of a single bond; 0 for the removal of a dative bond, which moves no charge:
    :func:`dative_bond_split_sites`). Compared atom by atom: element, isotope, number of radical electrons,
    number of hydrogens, and the formal charge (the charge before plus the declared change); then
    the bond set (``before`` minus the broken bonds), the sum of the bond orders at each atom of a
    Kekule drawing of each molecule, and the net charge.
    A bond listed twice, a bond that ``before`` does not have, or any RDKit error is False.
    PURE: read-only inspection of both molecules."""
    if before is None or after is None:
        return False
    try:
        count = before.GetNumAtoms()
        if after.GetNumAtoms() != count:
            return False
        delta = [0] * count
        broken = set()
        for metal_idx, other_idx in changed_bonds:
            key = frozenset((metal_idx, other_idx))
            if key in broken or before.GetBondBetweenAtoms(metal_idx, other_idx) is None:
                return False
            broken.add(key)
            delta[metal_idx] += moved_charge
            delta[other_idx] -= moved_charge
        for idx in range(count):
            old, new = before.GetAtomWithIdx(idx), after.GetAtomWithIdx(idx)
            if (old.GetAtomicNum() != new.GetAtomicNum() or old.GetIsotope() != new.GetIsotope()
                    or old.GetNumRadicalElectrons() != new.GetNumRadicalElectrons()
                    or old.GetTotalNumHs() != new.GetTotalNumHs()
                    or new.GetFormalCharge() != old.GetFormalCharge() + delta[idx]):
                return False

        def pairs(mol, skip):
            return {frozenset((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx())) for bd in mol.GetBonds()
                    if frozenset((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx())) not in skip}

        def bond_order_sums(mol, skip):
            # the sum of the bond orders at each atom of a Kekule drawing: an aromatic bond of one
            # perception (a pyrrolide that is aromatic once its nitrogen has no metal bond) and the
            # alternating bonds of another are one structure, and the sums, with the charges and
            # the hydrogens above, tell them from a drawing in which a bond order moved
            kekule = Chem.Mol(mol)
            Chem.Kekulize(kekule, clearAromaticFlags=True)
            sums = [0.0] * kekule.GetNumAtoms()
            for bd in kekule.GetBonds():
                if frozenset((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx())) in skip:
                    continue
                order = bd.GetBondTypeAsDouble()
                sums[bd.GetBeginAtomIdx()] += order
                sums[bd.GetEndAtomIdx()] += order
            return sums

        if pairs(before, broken) != pairs(after, set()):
            return False
        if bond_order_sums(before, broken) != bond_order_sums(after, set()):
            return False
        return Chem.GetFormalCharge(before) == Chem.GetFormalCharge(after)
    except (RuntimeError, ValueError):
        return False


def has_covalent_metal_carbon_bond(mol: "Chem.Mol") -> bool:
    """W8-P9 Task 9.2: True iff a TRUE metal atom (see
    ``_TRUE_METAL_SYMBOLS_FOR_VETO``) shares a DIRECT bond with a carbon atom
    (same connected fragment) — the structural definition of a
    organometallic compound (BB: "Organometallic compounds are
    compounds having at least one bond between one metal atom and one carbon
    atom"), scoped to metals that have NO legitimate alternate substitutive
    nomenclature system in this codebase.

    Deliberately excludes dot-separated ionic topologies (e.g. ferrocene
    `[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1`): RDKit ``GetBonds`` only enumerates
    bonds that exist in the graph, so a metal cation and an anionic Cp ring
    in SEPARATE fragments (no bond between them) correctly return False here
    — those compounds legitimately cascade to SALT@100 (ionic complex, no
    covalent M-C bond) and must keep doing so.

    Also deliberately excludes Group 13-16 metalloids (Si/Ge/Sn/Pb/B/As/Sb/
    Bi/Ga/In/Tl/Te/Po) even though they ARE metals/semimetals per
    ``is_metal_element`` — those have a legitimate alternate substitutive
    path / MONONUCLEAR_HYDRIDE / -04 heterocycle defer) that names
    the whole molecule via a DIFFERENT CFR class; flagging them here would
    false-positive-suppress correct names like 'trimethylarsane',
    'trimethylgallane', or '1-methyl-1H-silole'.

    Used as a source-level (no-OPSIN-required) veto: any compound for which
    this returns True IS a organometallic that must be named by the
    organometallic handler or fail closed — never silently renamed from a
    decomposed sub-fragment that drops the metal (the `C[Ti](Cl)(Cl)Cl` ->
    'methane' / `Cl[Pt]1(Cl)...` -> '...ole' leaks).

    PURE per internal notes: read-only GetBonds/GetBeginAtom/GetEndAtom/
    GetSymbol; no mol mutation.
    """
    if mol is None:
        return False
    for b in bonds_of(mol):
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        s1, s2 = a1.GetSymbol(), a2.GetSymbol()
        if s1 == 'C' and s2 in _TRUE_METAL_SYMBOLS_FOR_VETO:
            return True
        if s2 == 'C' and s1 in _TRUE_METAL_SYMBOLS_FOR_VETO:
            return True
    return False


def has_metal_coordination_bond(mol: "Chem.Mol") -> bool:
    """ (ChEBI 3-way, 2026-08-28): True iff a TRUE metal atom (the SAME
    ``_TRUE_METAL_SYMBOLS_FOR_VETO`` set as ``has_covalent_metal_carbon_bond``)
    is an endpoint of a DATIVE (coordination) bond -- a coordination
    compound whose metal binds via O/N/P dative bonds rather than a covalent
    M-C bond, so the C-bond veto above misses it. Confirmed leaks:
    Gd-DOTA (an MRI agent) -> 'ethanamide', Mo-dinitrogen bis-phosphine ->
    '(dimethylphosphanyl)benzene', an Fe siderophore -> a huge fatty-acyl
    diamide -- all silent structure-loss (the decomposition drops the metal and
    names a leftover ligand fragment; the RT/E1 gates miss it because the metal
    ends up in a discarded component).

    Scoped exactly like the covalent-C veto so it CANNOT false-positive on the
    in-scope classes:
      * metalloids (B/Si/Ge/Sn/As/Sb/Te/...) are excluded from the true-metal
        set -> boronic acids, silanes, arsanes stay nameable;
      * only DATIVE bonds count -> an ionic salt's lone metal cation
        (``[Na+]``, degree 0) has no bond, and a covalently-drawn metal
        carboxylate (a plain single M-O bond) is not dative, so neither is
        flagged -- salts remain in scope.
    PURE: read-only bond/atom inspection; no mol mutation."""
    if mol is None:
        return False
    from rdkit import Chem as _Chem
    _dative = (_Chem.BondType.DATIVE, _Chem.BondType.DATIVEONE,
               _Chem.BondType.DATIVEL, _Chem.BondType.DATIVER)
    for b in bonds_of(mol):
        if b.GetBondType() in _dative and (
                b.GetBeginAtom().GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO
                or b.GetEndAtom().GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO):
            return True
    return False


def detect_metallacycle(mol: "Chem.Mol") -> Optional["MetallacycleInfo"]:
    """W8-P9 Task 9.5: detect a metallacycle — a metal atom (Group
    2-12, per ``data.organometallics.METALLACYCLE_A_PREFIX``) that is itself
    a RING member.

    Scope THIS CYCLE (monocyclic, all-carbon backbone; narrow and
    conservative — fails closed, returns None, everywhere outside it):
      - exactly one metal ring atom (>1 -> polymetallic ring, out of scope);
      - the metal is in exactly ONE ring (fused/bridged/bicyclic metallacycles
        — e.g. the BB's own titanabicyclo[3.2.0]heptane example — are
        EXPLICITLY deferred per the plan's open question; a metal shared
        between 2+ rings returns None here);
      - ring size 4-8 (common metallacycle range);
      - every OTHER ring atom is carbon (mixed-heteroatom-backbone
        metallacycles, e.g. the BB's 1-sila-2-ferracyclopentane example, are
        out of scope this cycle);
      - the metal carries a formal charge of 0.

    Does NOT collide with the -04 Group-14/13 Hantzsch-Widman ring defer
    (``_GROUP_14_13_RING_DEFER``) — that set (Si/Ge/Sn/Pb/B) is disjoint from
    ``METALLACYCLE_A_PREFIX`` (Group 2-12) by construction.

    PURE per internal notes: read-only RingInfo/GetAtoms/GetBonds/GetNeighbors;
    no mol mutation.
    """
    from ..data.organometallics import METALLACYCLE_A_PREFIX

    if mol is None:
        return None
    metal_ring_atoms = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetSymbol() in METALLACYCLE_A_PREFIX and atom.IsInRing()
    ]
    if len(metal_ring_atoms) != 1:
        return None
    metal_idx = metal_ring_atoms[0]
    metal_atom = mol.GetAtomWithIdx(metal_idx)
    if metal_atom.GetFormalCharge() != 0:
        return None

    ri = mol.GetRingInfo()
    atom_rings = [r for r in ri.AtomRings() if metal_idx in r]
    if len(atom_rings) != 1:
        return None  # metal in 0 or >1 rings -- fused/bicyclic, deferred
    ring = atom_rings[0]
    if not (4 <= len(ring) <= 8):
        return None
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring if i != metal_idx):
        return None  # mixed-heteroatom-backbone metallacycle -- out of scope

    ring_set = set(ring)
    exocyclic: List[Tuple[int, ...]] = []
    for bond in metal_atom.GetBonds():
        other = bond.GetOtherAtomIdx(metal_idx)
        if other in ring_set:
            continue
        exocyclic.append((other,))

    return MetallacycleInfo(
        metal_atom_idx=metal_idx,
        ring_atom_indices=tuple(ring),
        exocyclic_atom_indices=tuple(exocyclic),
    )


def enumerate_metal_ligand_groups(mol: "Chem.Mol") -> Tuple[LigandGroup, ...]:
    """For each metal atom in mol, partition coordinated atoms into ligand groups.

    STUB BODY (Plan-02): returns  always. Plan-03 implements per
    RESEARCH lines 549-567.

    Algorithm (Plan-03 implementation):
    1. Identify all metal atoms via is_metal_element(atom.GetSymbol).
    2. For each metal: collect bond-distance-1 neighbors (ligand-shell).
    3. For each ligand-shell atom: BFS over non-metal bonds to find
       contiguous ligand component.
    4. Each component → LigandGroup with hapticity from compute_hapticity.

    PURE per internal notes. O(V+E) per metal.
    """
    # Plan-03 implementation lands here per RESEARCH + AUDIT
    return ()


__all__ = [
    'METAL_ELEMENT_SYMBOLS',
    'is_metal_element',
    'LigandGroup',
    'MetalComplex',
    'MetallacycleInfo',
    'detect_metal_complex',
    'compute_hapticity',
    'enumerate_metal_ligand_groups',
    'has_covalent_metal_carbon_bond',
    'detect_metallacycle',
]
