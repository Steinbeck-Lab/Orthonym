"""
Organometallic complex detection + hapticity perception (Phase 161 v19).

This module provides RDKit-graph-based detection of metal-organometallic
complexes per IUPAC 2013 Blue Book §P-69 + IUPAC Red Book §IR-10 + Salzer
1999 IUPAC Recommendations.

All functions are PREDICATE-PURE per CONTEXT D-12 + Phase 158 D-26 + Phase
160 D-25 hard invariant:
- NO mol mutation (no SanitizeMol, no UpdatePropertyCache, no
  AssignStereochemistry, no any other RDKit mutating method).
- NO module-global state read/write.
- NO exception swallowing (RDKit exceptions propagate; ValueError raised
  by compute_hapticity is caught at the handler layer per CONTEXT D-12).

The integrity test ``test_side_effect_inventory_is_empty`` parametrized
over ``list(StoutClass)`` at ``tests/unit/routing/test_dispatch_table.py``
will auto-extend to ORGANOMETALLIC after Plan-02 ships and assert ``()``
for the new entry.

Phase 161 (v19 first scope-expansion phase per ADR-19-07).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple
from rdkit import Chem


# Allowlist for is_metal_element() per RESEARCH §3.1.
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
    # Lanthanides + Actinides (Phase 161.X scope-deferred but allowlist anyway)
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
})


# WSD-04 (RING-06): Group-14/13 metalloids that, when a ring member of a covalent
# heterocycle, are a Hantzsch-Widman ring parent (silolane/silole/borole/stannole),
# NOT an organometallic complex. detect_metal_complex defers these to the
# heterocycle path. Scope per the requirement: Si/Ge/Sn/Pb (Group 14) + B (Group 13).
# Deliberately EXCLUDES P (a separate phosphine-handler concern) and Al/Ga/In/Tl
# (kept under organometallic handling — out of WSD-04 scope).
_GROUP_14_13_RING_DEFER: FrozenSet[str] = frozenset({'Si', 'Ge', 'Sn', 'Pb', 'B'})


# P-69.5 metal classification for mixed-metal covalent compounds:
#   (1) metals of Groups 1 through 12  -> central-atom eligible;
#   (2) metals of Groups 13 through 16 -> cited as a substituent group.
# When exactly ONE class-1 metal and >=1 class-2 metalloid are present in a single
# covalent component, the class-1 metal is the central atom and the class-2
# metalloid sits inside a ligand (P-69.5.2: e.g. C6H5-Hg-C6H4-Sb(C6H5)2 ->
# [4-(diphenylstibanyl)phenyl](phenyl)mercury). _CLASS_2 = Groups 13-16 metals/
# metalloids; _CLASS_1 = every other (Groups 1-12) metal in METAL_ELEMENT_SYMBOLS.
_CLASS_2_METALLOIDS: FrozenSet[str] = frozenset({
    'B', 'Al', 'Ga', 'In', 'Tl',        # Group 13
    'Si', 'Ge', 'Sn', 'Pb',             # Group 14
    'As', 'Sb', 'Bi',                   # Group 15
    'Te', 'Po',                         # Group 16
})
_CLASS_1_METALS: FrozenSet[str] = METAL_ELEMENT_SYMBOLS - _CLASS_2_METALLOIDS


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
    """W8-P9 Task 9.5 (P-69.4): a metal atom that is itself a RING member.

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

    PURE per CONTEXT D-12: no mol mutation; only reads via GetAtoms /
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
    for atom in mol.GetAtoms():
        if is_metal_element(atom.GetSymbol()):
            metal_atom_indices.append(atom.GetIdx())
            formal_charges.append(atom.GetFormalCharge())

    if not metal_atom_indices:
        return None

    # === Tier-1 / Tier-2 topology: dot-separated multi-component ===
    # Frags structure: 1 metal frag (single atom) + N ligand frags
    # (Cp anions for Tier-1, CO ligands for Tier-2).
    frags = Chem.GetMolFrags(mol, asMols=False)

    # P-69.5.2 mixed-class single-centred complex: EXACTLY ONE class-1 (Groups 1-12)
    # central metal + >=1 class-2 (Groups 13-16) metalloid, all in ONE covalent
    # component. The class-1 metal is the sole central atom; the class-2 metalloid
    # is named as a substituent inside a ligand. This is NOT a multinuclear bridge
    # (Phase 161.3 deferred). class-2/class-2 catenated hydrides (0 class-1 metals)
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

    # Phase 161.3 deferred: multinuclear bridged complexes (but NOT the mixed-class
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
        # For a P-69.5.2 mixed-class compound the class-1 metal is the sole central
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
        # WSD-04 (RING-06): a single ring-member Group-14/13 metalloid
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

    Returns None if the topology doesn't match σ-bonded (e.g., aromatic
    π-bonded ligand attached to metal). Pure per CONTEXT D-12.
    """
    metal_atom = mol.GetAtomWithIdx(metal_idx)
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

        # Skip if connecting via a non-single bond (aromatic / double / triple)
        # — those are π-bonded, Tier-4 territory. Aryl σ-bonds to metal are
        # single bonds; aromatic-ring atoms inside the ring are aromatic but
        # the M-C bond itself is single.
        if neighbor_bond.GetBondTypeAsDouble() > 1.5:
            return None

        # Walk the organic ligand subgraph via BFS, not crossing the central
        # metal; class-2 metalloids in ``walk_through`` are traversed (they are
        # part of this ligand, e.g. the Sb of a diphenylstibanyl-phenyl ligand).
        ligand_atoms = _collect_organic_component(
            mol, neighbor_idx, metal_idx, walk_through)
        # Refuse to collect atoms already in another ligand (e.g., a bridging
        # carbon would belong to two metals — Phase 161.3 territory)
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
        return None
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

    Plan-03 implementation per CONTEXT D-05 hybrid:
    1. SMARTS-template fast path: consult LIGAND_ETA_DEFAULTS from
       data/organometallics.py; on hit, return catalog value.
    2. Graph-walk fallback (added in 03-04 Tier-4 commit): BFS over
       ligand_atom_indices counting contiguous π-system atoms.
    3. Edge case σ-bonded (single atom or no π-system): return 1.

    PURE per CONTEXT D-12.
    """
    # Lazy import to avoid circular dependency
    from ..data.organometallics import LIGAND_ETA_DEFAULTS

    if len(ligand_atom_indices) == 0:
        raise ValueError(f"empty ligand_atom_indices for metal {metal_atom_idx}")

    # Single-atom ligand: σ-bonded hapticity 1
    if len(ligand_atom_indices) == 1:
        return 1

    # Build the ligand subgraph as a separate mol (PURE per D-12: submol only)
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
# actinides), per BB P-69.0's own three-way split: "(1) elements of Groups 1
# and 2; (2) elements of Groups 3-12 (the transition metals); (3) elements
# of Groups 13-16". Deliberately EXCLUDES the Group 13-16 metalloids (B, Al,
# Ga, In, Tl, Si, Ge, Sn, Pb, As, Sb, Bi, Te, Po): those ALWAYS have a
# legitimate alternate SUBSTITUTIVE nomenclature system (P-68) that correctly
# names the WHOLE molecule via a different CFR class —
# MONONUCLEAR_HYDRIDE (trimethylarsane/trimethylgallane/trimethylindigane/
# trimethylthallane), the WSD-04 Hantzsch-Widman ring defer (silole/borole/
# stannole), FREE_HOMONUCLEAR_G14_HYDRIDE, DINUCLEAR_HYDRIDE, etc. — so a
# "class_id != organometallic" outcome for one of THOSE elements is not an
# atom-drop leak; it is the correct alternate path. Restricting the veto's
# metal scope to true metals (which have NO alternate substitutive system in
# this codebase) makes the veto SAFE: a true-metal-carbon bond can only be
# legitimately named via P-69 additive/coordination nomenclature, so any
# other outcome for it genuinely is a structure-loss leak.
_TRUE_METAL_SYMBOLS_FOR_VETO: FrozenSet[str] = METAL_ELEMENT_SYMBOLS - frozenset({
    'B', 'Al', 'Ga', 'In', 'Tl',       # Group 13
    'Si', 'Ge', 'Sn', 'Pb',            # Group 14
    'As', 'Sb', 'Bi',                  # Group 15
    'Te', 'Po',                        # Group 16
})


def has_covalent_metal_carbon_bond(mol: "Chem.Mol") -> bool:
    """W8-P9 Task 9.2: True iff a TRUE metal atom (see
    ``_TRUE_METAL_SYMBOLS_FOR_VETO``) shares a DIRECT bond with a carbon atom
    (same connected fragment) — the structural definition of a P-69
    organometallic compound (BB P-69.0: "Organometallic compounds are
    compounds having at least one bond between one metal atom and one carbon
    atom"), scoped to metals that have NO legitimate alternate substitutive
    nomenclature system in this codebase.

    Deliberately excludes dot-separated ionic topologies (e.g. ferrocene
    `[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1`): RDKit ``GetBonds()`` only enumerates
    bonds that exist in the graph, so a metal cation and an anionic Cp ring
    in SEPARATE fragments (no bond between them) correctly return False here
    — those compounds legitimately cascade to SALT@100 (ionic complex, no
    covalent M-C bond) and must keep doing so.

    Also deliberately excludes Group 13-16 metalloids (Si/Ge/Sn/Pb/B/As/Sb/
    Bi/Ga/In/Tl/Te/Po) even though they ARE metals/semimetals per
    ``is_metal_element`` — those have a legitimate alternate substitutive
    path (P-68 / MONONUCLEAR_HYDRIDE / WSD-04 heterocycle defer) that names
    the whole molecule via a DIFFERENT CFR class; flagging them here would
    false-positive-suppress correct names like 'trimethylarsane',
    'trimethylgallane', or '1-methyl-1H-silole'.

    Used as a source-level (no-OPSIN-required) veto: any compound for which
    this returns True IS a P-69 organometallic that must be named by the
    organometallic handler or fail closed — never silently renamed from a
    decomposed sub-fragment that drops the metal (the `C[Ti](Cl)(Cl)Cl` ->
    'methane' / `Cl[Pt]1(Cl)...` -> '...ole' leaks).

    PURE per CONTEXT D-12: read-only GetBonds/GetBeginAtom/GetEndAtom/
    GetSymbol; no mol mutation.
    """
    if mol is None:
        return False
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        s1, s2 = a1.GetSymbol(), a2.GetSymbol()
        if s1 == 'C' and s2 in _TRUE_METAL_SYMBOLS_FOR_VETO:
            return True
        if s2 == 'C' and s1 in _TRUE_METAL_SYMBOLS_FOR_VETO:
            return True
    return False


def detect_metallacycle(mol: "Chem.Mol") -> Optional["MetallacycleInfo"]:
    """W8-P9 Task 9.5 (P-69.4): detect a metallacycle — a metal atom (Group
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

    Does NOT collide with the WSD-04 Group-14/13 Hantzsch-Widman ring defer
    (``_GROUP_14_13_RING_DEFER``) — that set (Si/Ge/Sn/Pb/B) is disjoint from
    ``METALLACYCLE_A_PREFIX`` (Group 2-12) by construction.

    PURE per CONTEXT D-12: read-only RingInfo/GetAtoms/GetBonds/GetNeighbors;
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

    STUB BODY (Plan-02): returns () always. Plan-03 implements per
    RESEARCH §3.2 lines 549-567.

    Algorithm (Plan-03 implementation):
    1. Identify all metal atoms via is_metal_element(atom.GetSymbol()).
    2. For each metal: collect bond-distance-1 neighbors (ligand-shell).
    3. For each ligand-shell atom: BFS over non-metal bonds to find
       contiguous ligand component.
    4. Each component → LigandGroup with hapticity from compute_hapticity.

    PURE per CONTEXT D-12. O(V+E) per metal.
    """
    # Plan-03 implementation lands here per RESEARCH §3.2 + AUDIT § 5
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
