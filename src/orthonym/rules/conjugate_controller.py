"""Class-agnostic conjugate-fragment classifier (Phase 182, WSC-03, D-02).

A NEW standalone, class-agnostic primitive: given a molecule, a scaffold attachment
atom, the linker atom (``first_idx``) reached through it, and the scaffold atom set, it
classifies the fragment past the linker as a **sulfate ester**, **mono-phosphate
ester**, or **glycosyl/uronyl** conjugate and returns the functional-class word/head
plus the consumed-atom set. Returns ``None`` (fail-closed) for anything else.

This is the cross-class deliverable (WSC-03 crit #2): the signature is
``classify_conjugate(mol, attach_idx, first_idx, scaffold_atoms)`` with NO
steroid-specific branch, so the glycoside (Phase 176) and lipid (Phase 180) paths can
call the SAME primitive later. This phase *wires* it into the NP subsystem (182-02);
the *logic* here is class-agnostic.

Charge -> word is derived **in place** from the protonation/ionisation state of the
acid centre on the ORIGINAL molecule (D-04) — never neutralize-then-rename (the WS-E
failure mode), never a per-molecule hardcode:
  -OSO2[O-]  -> "sulfate"            -OSO2OH    -> "hydrogen sulfate"
  -OPO(OH)2  -> "dihydrogen phosphate"  mono-anion -> "hydrogen phosphate"
  di-anion   -> "phosphate"
  (BB P-65.6.3.3.5 @31935 partial esters/salts; P-102.5.6.1.2 @53199 phosphate
   ionisation; word examples @35968 / @35940 / @41005.)

The glycoside branch caps the broken glycosidic bond at the ATOM level
(``Chem.FragmentOnBonds`` + restore the anomeric -OH + ``Chem.MolToSmiles``) so the
capped fragment canonicalizes to a ``URONIC_ACID_NAMES`` key — NO string surgery on the
sugar head (RESEARCH Open Q2 RESOLVED). The uronic head form comes from the explicit
``data.sugar_names.uronic_glycoside_head`` map (BB P-102.5.6.6.4.2 @53789).

Root-cause-only (CLAUDE.md): no postprocessor, no regex on any existing name string,
no neutralize-then-rename, no per-molecule hardcode. All logic is RDKit atom/bond
walks + dict lookups + set math; the function is pure (no global state, no mol
mutation — the NP dispatch calls the path twice, Pitfall 5).
"""

from typing import Dict, List, Optional, Set

from rdkit import Chem

from orthonym.data.sugar_names import (
    lookup_sugar,
    recognize_sugar_skeleton,
    sugar_to_glycoside_class_name,
    uronic_glycoside_head,
)

# Charge -> word, keyed on the number of *protonated* terminal acidic oxygens (D-04).
SULFATE_WORD = {1: "hydrogen sulfate", 0: "sulfate"}
PHOSPHATE_WORD = {2: "dihydrogen phosphate", 1: "hydrogen phosphate", 0: "phosphate"}


def _terminal_acid_oxygens(mol, central_idx: int, linker_o_idx: int):
    """Count terminal acidic oxygens on the S/P centre.

    O atoms bonded to the central S/P, excluding the linker-O back to the scaffold
    and any doubly-bonded ``=O``. Each remaining O is protonated (a free -OH:
    ``GetTotalNumHs() >= 1``) or anionic (``GetFormalCharge() < 0``). A bare O written
    without explicit H or charge is treated as anionic.

    Returns ``(n_protonated, n_anionic)``. Mirrors the in-place carbonyl-detection
    walk in ``natural_products._find_ester_decorations`` (lines 1466-1481).
    """
    central = mol.GetAtomWithIdx(central_idx)
    prot = anion = 0
    for nbr in central.GetNeighbors():
        if nbr.GetIdx() == linker_o_idx or nbr.GetAtomicNum() != 8:
            continue
        b = mol.GetBondBetweenAtoms(central_idx, nbr.GetIdx())
        if b.GetBondType() == Chem.BondType.DOUBLE:
            continue  # the =O, not an acidic OH/O-
        if nbr.GetFormalCharge() < 0:
            anion += 1
        elif nbr.GetTotalNumHs() >= 1:
            prot += 1
        else:
            anion += 1  # bare O without explicit H/charge -> treat as anionic
    return prot, anion


def _fragment_atoms(mol, first_idx: int, exclude: Set[int]) -> Set[int]:
    """BFS over the conjugate fragment from ``first_idx``, excluding scaffold atoms.

    Collects ``{first_idx} | {...fragment...}`` — every atom reachable from the linker
    atom without crossing back into ``exclude`` (the scaffold + attach atom).
    """
    seen: Set[int] = set()
    stack = [first_idx]
    while stack:
        idx = stack.pop()
        if idx in seen or idx in exclude:
            continue
        seen.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            n = nbr.GetIdx()
            if n not in seen and n not in exclude:
                stack.append(n)
    return seen


def _other_neighbor(mol, atom_idx: int, skip: int) -> Optional[int]:
    """The single heavy neighbor of ``atom_idx`` other than ``skip`` (or None)."""
    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if nbr.GetIdx() == skip:
            continue
        return nbr.GetIdx()
    return None


def _has_double_bonded_oxygen(mol, central_idx: int) -> bool:
    central = mol.GetAtomWithIdx(central_idx)
    for nbr in central.GetNeighbors():
        if nbr.GetAtomicNum() != 8:
            continue
        b = mol.GetBondBetweenAtoms(central_idx, nbr.GetIdx())
        if b.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def _count_double_bonded_oxygens(mol, central_idx: int) -> int:
    n = 0
    central = mol.GetAtomWithIdx(central_idx)
    for nbr in central.GetNeighbors():
        if nbr.GetAtomicNum() != 8:
            continue
        b = mol.GetBondBetweenAtoms(central_idx, nbr.GetIdx())
        if b.GetBondType() == Chem.BondType.DOUBLE:
            n += 1
    return n


def _has_second_phosphorus(mol, central_idx: int, fragment: Set[int]) -> bool:
    """True if a SECOND P (or a P-O-P bridge) is present in the fragment.

    Multi-phosphate (di/triphosphate) is out of scope this phase (defer to 183/184).
    """
    for a in fragment:
        if a == central_idx:
            continue
        if mol.GetAtomWithIdx(a).GetAtomicNum() == 15:
            return True
    return False


def _extract_capped_sugar(mol, anomeric_idx: int, linker_o_idx: int) -> Optional[str]:
    """Cap the broken glycosidic bond at the atom level and return canonical SMILES.

    Cleaves the anomeric-C — linker-O bond (``Chem.FragmentOnBonds``), keeps the
    sugar-side fragment, and restores the anomeric -OH by replacing the dummy with an
    O carrying one explicit H. Returns ``Chem.MolToSmiles`` of the sanitized,
    H-removed sugar — which canonicalizes to a ``URONIC_ACID_NAMES`` key for
    glucuronides (RESEARCH Open Q2). NO string surgery.

    Mirrors the RDKit primitives in ``decomposition/fragment_capping.cleave_and_cap``;
    kept inline so the classifier stays self-contained (D-02 — no import from
    ``decomposition/``).
    """
    bond = mol.GetBondBetweenAtoms(anomeric_idx, linker_o_idx)
    if bond is None:
        return None
    try:
        frag = Chem.FragmentOnBonds(
            mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0, 0)]
        )
    except Exception:
        return None
    mapping: List = []
    try:
        frags = Chem.GetMolFrags(
            frag, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping
        )
    except Exception:
        return None
    sugar = None
    for fm, mp in zip(frags, mapping):
        if anomeric_idx in mp:
            sugar = fm
            break
    if sugar is None:
        return None
    rw = Chem.RWMol(sugar)
    for at in rw.GetAtoms():
        if at.GetAtomicNum() == 0:  # the dummy from the cleaved bond
            at.SetAtomicNum(8)
            at.SetNoImplicit(False)
            at.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        try:
            Chem.SanitizeMol(
                rw, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
        except Exception:
            return None
    try:
        clean = Chem.RemoveHs(rw)
        return Chem.MolToSmiles(clean)
    except Exception:
        return None


def classify_conjugate(
    mol, attach_idx: int, first_idx: int, scaffold_atoms: Set[int]
) -> Optional[Dict]:
    """Classify a conjugate fragment reached through a scaffold heteroatom linker.

    Args:
        mol: the RDKit Mol (read-only; not mutated).
        attach_idx: the scaffold atom the fragment attaches to.
        first_idx: the linker atom (an ester-O, NOT an -OH) past ``attach_idx``.
        scaffold_atoms: the set of scaffold atom indices to exclude from the walk.

    Returns:
        ``{"kind", "word", "all_atoms", "linker_kind"}`` for a sulfate / mono-phosphate
        / glycosyl(uronyl) fragment, else ``None`` (fail-closed).
    """
    first_atom = mol.GetAtomWithIdx(first_idx)
    # The linker must be an ester-type O with no H (an -OH is a hydroxyl, not a linker).
    if first_atom.GetAtomicNum() != 8 or first_atom.GetTotalNumHs() > 0:
        return None

    exclude: Set[int] = set(scaffold_atoms) | {attach_idx}
    fragment = _fragment_atoms(mol, first_idx, exclude)

    # The atom past the linker-O determines the branch.
    central_idx = _other_neighbor(mol, first_idx, attach_idx)
    if central_idx is None:
        return None
    central = mol.GetAtomWithIdx(central_idx)
    z = central.GetAtomicNum()

    # ---- SULFATE: linker-O -> S with two =O, classify the remaining terminal O ----
    if z == 16:  # sulfur
        if _count_double_bonded_oxygens(mol, central_idx) != 2:
            return None
        prot, _anion = _terminal_acid_oxygens(mol, central_idx, first_idx)
        word = SULFATE_WORD.get(prot)
        if word is None:
            return None
        return {
            "kind": "sulfate",
            "word": word,
            "all_atoms": fragment,
            "linker_kind": "O",
        }

    # ---- PHOSPHATE (mono only): linker-O -> P with one =O, two terminal O slots ----
    if z == 15:  # phosphorus
        if _count_double_bonded_oxygens(mol, central_idx) != 1:
            return None
        if _has_second_phosphorus(mol, central_idx, fragment):
            return None  # di/triphosphate or P-O-P bridge -> out of scope (183/184)
        prot, _anion = _terminal_acid_oxygens(mol, central_idx, first_idx)
        word = PHOSPHATE_WORD.get(prot)
        if word is None:
            return None
        return {
            "kind": "phosphate",
            "word": word,
            "all_atoms": fragment,
            "linker_kind": "O",
        }

    # ---- GLYCOSIDE: linker-O -> anomeric C of a recognized sugar ring ----
    if z == 6 and central.IsInRing():
        # anomeric C is bonded to a ring-O (the ring oxygen of the sugar pyranose/furanose)
        has_ring_o = any(
            rn.GetAtomicNum() == 8 and rn.IsInRing()
            for rn in central.GetNeighbors()
        )
        if not has_ring_o:
            return None
        canon = _extract_capped_sugar(mol, central_idx, first_idx)
        if not canon:
            return None
        tup = lookup_sugar(canon)
        if tup is None:
            sugar_mol = Chem.MolFromSmiles(canon)
            tup = recognize_sugar_skeleton(sugar_mol) if sugar_mol is not None else None
        if tup is None:
            return None
        anomer, config, base = tup
        if "urono" in base:
            head = uronic_glycoside_head(anomer, config, base)
        else:
            head = sugar_to_glycoside_class_name(anomer, config, base)
        if not head:
            return None
        return {
            "kind": "glycoside",
            "word": head,
            "all_atoms": fragment,
            "linker_kind": "O",
        }

    # Fail-closed for anything else.
    return None
