"""
Adduct / solvate / hydrate nomenclature — Blue Book P-14.8 (Wave-2 P0A).

P-14.8.1: "Names are formed by citing the names of individual compounds in
the order of the formula connected by long (em) dashes (—). The
proportions of components are indicated after the name by an arabic number
separated by a solidus from other numbers; arabic numbers and the solidus
are placed in parentheses, separated from the name by a space."

P-14.8.2: "organic components in order as described in P-14.8.1, inorganic
components...; water (if present), is cited last." General nomenclature:
"hydrates may be named by adding the word 'hydrate' to the name preceded by
an appropriate numerical prefix... Terms such as 'hemi' and 'sesqui' are
also used."

Fail-closed: `name_adduct` returns None unless EVERY component fragment is
fully nameable and every single-heavy-atom fragment is a recognized
inorganic component. Scope: ALL-NEUTRAL fragment sets only — charged
multi-fragment input is owned by the salt/ion routing (dispatch priority
< 800) and never reaches this module.
"""

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

EM_DASH = "—"

# P-14.8.2 single-heavy-atom inorganic components. Water is cited last; the
# hydracids use the binary names the Blue Book's own P-14.8.2 examples use
# ("3-[(2S)-1-methylpyrrolidin-2-yl]pyridine—hydrogen chloride (1/1)",
# BlueBookV2.md line 4677). Any OTHER single-atom fragment (bare metals,
# ammonia is deliberately excluded) makes the input NOT an adduct for this
# namer -> decline (fail-closed).
SINGLE_ATOM_COMPONENT_NAMES: Dict[str, str] = {
    "O": "water",
    "F": "hydrogen fluoride",
    "Cl": "hydrogen chloride",
    "Br": "hydrogen bromide",
    "I": "hydrogen iodide",
}


def split_components(mol) -> Optional[List[Tuple[str, int]]]:
    """Split a multi-fragment mol into deduped (canonical_smiles, count).

    Returns None for single-fragment input or when RDKit cannot split /
    sanitize the fragments (fail-closed).
    """
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return None
    if len(frags) < 2:
        return None
    counts: Dict[str, int] = {}
    order: List[str] = []
    for frag in frags:
        smi = Chem.MolToSmiles(frag, canonical=True)
        if smi not in counts:
            counts[smi] = 0
            order.append(smi)
        counts[smi] += 1
    return [(smi, counts[smi]) for smi in order]


def _name_component(frag_smi: str, style: str) -> Optional[str]:
    """Name ONE component fragment, or None (fail-closed).

    Single-heavy-atom fragments come ONLY from the P-14.8.2 table above.
    Multi-atom fragments go through the full single-component pipeline via
    a FRESH Orthonym instance (audit §3.3 RL-4 fresh-instance pattern,
    same as routing/dispatch_table._handle_multi_component_neutral), so the
    per-fragment OPSIN validity gate stays ON in production.
    """
    frag_mol = Chem.MolFromSmiles(frag_smi)
    if frag_mol is None:
        return None
    if Chem.GetFormalCharge(frag_mol) != 0:
        return None  # charged fragments belong to the salt/ion router
    if frag_mol.GetNumHeavyAtoms() == 1:
        return SINGLE_ATOM_COMPONENT_NAMES.get(
            Chem.MolToSmiles(frag_mol, canonical=True))
    from orthonym.namer import Orthonym  # lazy: avoid import cycle
    try:
        name = Orthonym(style=style).name(frag_smi)
    except Exception:
        return None
    if not name or not isinstance(name, str) or name.startswith("unknown"):
        return None
    if "not supported" in name:
        return None  # descriptive refusal placeholders are not names
    # Fail-closed OPSIN-parseability gate on the COMPONENT name itself.
    # In production the per-instance validity gate already rejects
    # OPSIN-unparseable names (returns 'unknown...' above); but the unit
    # suite disables that gate module-wide (conftest
    # _disable_opsin_validity_gate_for_tests), which would otherwise let a
    # semantically-wrong-but-nonempty component name (e.g. the HEAD name
    # '2-amino-1-anilinoethanamide' for NCC(=O)Nc1ccc(OCC)cc1, which OPSIN
    # cannot parse) propagate into an adduct name. Re-assert parseability
    # here so the adduct assembler is self-contained and fails closed on an
    # unparseable component in EVERY context. Fail-OPEN on 'unavailable'
    # (no JAR / transient OPSIN error) exactly as the production gate does.
    from orthonym.namer import _validity_gate_status
    if _validity_gate_status(name) == "rejected":
        return None
    return name


def _component_bucket(frag_mol, frag_smi: str) -> int:
    """P-14.8.1/P-14.8.2 citation buckets: 0 organic, 1 inorganic, 2 water."""
    if frag_smi == "O":
        return 2  # "water (if present), is cited last" (P-14.8.2)
    if any(a.GetAtomicNum() == 6 for a in frag_mol.GetAtoms()):
        return 0  # "organic compounds precede inorganic compounds" (P-14.8.1)
    return 1


def component_sort_key(frag_smi: str) -> Tuple[int, int, int, str]:
    """Deterministic P-14.8 citation order for one component.

    (bucket, P-41 seniority index, -heavy_atoms, canonical_smiles):
    organic components ordered by the seniority of the class of their
    principal characteristic group (P-14.8.1: "cited in the order of
    seniority of classes (see P-41)"); components without a suffix-capable
    PCG (hydrocarbons, N-heterocycles-as-π-bases) rank after all
    PCG-bearing organics; ties by descending size then canonical SMILES —
    reproduces every P-14.8 Blue Book example (coronene—trinitrobenzene
    big-first; benzene—pyridine resolved form).
    """
    frag_mol = Chem.MolFromSmiles(frag_smi)
    if frag_mol is None:  # unreachable behind split_components; belt+braces
        return (3, 0, 0, frag_smi)
    bucket = _component_bucket(frag_mol, frag_smi)
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import SENIORITY_ORDER, get_principal_group
    seniority = len(SENIORITY_ORDER)
    if bucket == 0:
        try:
            pg, _ = get_principal_group(frag_mol, detect_functional_groups(frag_mol))
            if pg is not None and pg in SENIORITY_ORDER:
                seniority = SENIORITY_ORDER.index(pg)
        except Exception:
            pass  # no PCG -> ranks after PCG-bearing organics
    return (bucket, seniority, -frag_mol.GetNumHeavyAtoms(), frag_smi)
