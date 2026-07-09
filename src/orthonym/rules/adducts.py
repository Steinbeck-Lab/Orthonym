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


# P-14.8.2: "an appropriate numerical prefix, such as 'mono','di', 'tri'"
_HYDRATE_MULTIPLIERS = {
    1: "mono", 2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}


def _hydrate_word_form(named: List[Tuple[str, int]],
                       water_index: int) -> Optional[str]:
    """General-nomenclature '<components> <prefix>hydrate', or None.

    Defined ONLY when every non-water component shares one count p; the
    prefix encodes the reduced water:parent ratio w/p — n/1 -> mono/di/
    tri/... , 1/2 -> hemi, 3/2 -> sesqui (P-14.8.2 line 4657; BB line 4686
    pairs (2/2/3) with 'sesquihydrate'). Anything else returns None and
    the caller emits the always-valid proportion notation instead.
    """
    from math import gcd
    water_count = named[water_index][1]
    others = [nc for i, nc in enumerate(named) if i != water_index]
    if not others:
        return None  # water-only input is not a hydrate of anything
    parent_counts = {count for _name, count in others}
    if len(parent_counts) != 1:
        return None
    parent_count = parent_counts.pop()
    g = gcd(water_count, parent_count)
    w, p = water_count // g, parent_count // g
    if p == 1:
        prefix = _HYDRATE_MULTIPLIERS.get(w)
    elif p == 2 and w == 1:
        prefix = "hemi"
    elif p == 2 and w == 3:
        prefix = "sesqui"
    else:
        prefix = None
    if prefix is None:
        return None
    base = EM_DASH.join(name for name, _count in others)
    return f"{base} {prefix}hydrate"


def _assemble_adduct_name(named: List[Tuple[str, int]]) -> str:
    """P-14.8.1: names joined by em-dash; proportions '(n/m/...)' appended
    'separated from the name by a space'. Proportions are ALWAYS cited in
    the PIN form, including (1/1) (BB: 'benzene—pyridine (1/1)')."""
    names = EM_DASH.join(name for name, _count in named)
    proportions = "/".join(str(count) for _name, count in named)
    return f"{names} ({proportions})"


def name_adduct(mol, canonical_smiles: Optional[str] = None,
                style: str = "pin") -> Optional[str]:
    """Name an all-neutral multi-component input per P-14.8, or None.

    Fail-closed refusals (return None; the dispatch cascade then falls
    through to the honest 'unknown organic compound'):
      * fewer than 2 DISTINCT components (identical-only sets are not
        adducts — the dispatch handler keeps the frozen space-join there);
      * no multi-atom component at all;
      * any single-heavy-atom fragment outside SINGLE_ATOM_COMPONENT_NAMES
        (bare metals -> organometallic routing, never swallowed here);
      * any charged fragment (salt/ion routing owns charged input);
      * ANY component the single-component pipeline cannot name.
    """
    components = split_components(mol)
    if components is None or len(components) < 2:
        return None
    frag_mols = {smi: Chem.MolFromSmiles(smi) for smi, _ in components}
    if any(fm is None for fm in frag_mols.values()):
        return None
    if not any(fm.GetNumHeavyAtoms() >= 2 for fm in frag_mols.values()):
        return None
    for smi, fm in frag_mols.items():
        if Chem.GetFormalCharge(fm) != 0:
            return None
        if (fm.GetNumHeavyAtoms() == 1
                and smi not in SINGLE_ATOM_COMPONENT_NAMES):
            return None
    ordered = sorted(components, key=lambda t: component_sort_key(t[0]))
    named: List[Tuple[str, int]] = []
    for smi, count in ordered:
        component_name = _name_component(smi, style)
        if component_name is None:
            return None  # fail-closed: never drop or placeholder a component
        named.append((component_name, count))
    if style == "general":
        water_indices = [i for i, (smi, _c) in enumerate(ordered)
                         if smi == "O"]
        if water_indices:
            word = _hydrate_word_form(named, water_indices[0])
            if word is not None:
                return word
    return _assemble_adduct_name(named)
