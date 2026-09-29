"""
Adduct / solvate / hydrate nomenclature — Blue Book (Wave-2 P0A).

: "Names are formed by citing the names of individual compounds in
the order of the formula connected by long (em) dashes (—). The
proportions of components are indicated after the name by an arabic number
separated by a solidus from other numbers; arabic numbers and the solidus
are placed in parentheses, separated from the name by a space."

: "organic components in order as described in, inorganic
components...; water (if present), is cited last." General nomenclature:
"hydrates may be named by adding the word 'hydrate' to the name preceded by
an appropriate numerical prefix... Terms such as 'hemi' and 'sesqui' are
also used."

Fail-closed: `name_adduct` returns None unless EVERY component fragment is
fully nameable and every single-heavy-atom fragment is a recognized
inorganic component. Scope: ALL-NEUTRAL fragment sets only — charged
multi-fragment input is owned by the salt/ion routing (dispatch priority
< 800) and never reaches this module.

Breadth Job 2 (best-effort tier only): a disconnected drawing with a metal in it
('CC(N)C(=O)O.CC(N)C(=O)O.[Ni]', six '[C-]#N' with '[Fe+2]') is named as a
mixed organic - inorganic adduct of its components -- a metal atom by its element
name, a metal cation with its charge number, a metal halide or oxide by an additive
name -- and ships only when OPSIN reads the name back to exactly the drawn structure.
Such a name is never a PIN (the Blue Book).
"""

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

EM_DASH = "—"

# Single-heavy-atom NEUTRAL molecular components of a adduct, keyed by
# the fragment's element symbol and mapped to the component's own name. Water
# is cited last; the hydracids use the binary names the Blue Book's own
# examples use ("3-[(2S)-1-methylpyrrolidin-2-yl]pyridine—hydrogen
# chloride (1/1)", the Blue Book line 4677).
#
# The nonmetal hydrides below (methane/hydrogen sulfide/phosphane) are genuine
# neutral molecular species that occur as adduct partners in the corpus; each
# name was verified OPSIN-parseable in em-dash adduct notation
# ("benzene—methane (1/1)" etc.). Without them a row whose only single-atom
# component is a bare C/S/P declined outright -- a table-miss-degrades-to-
# refusal defect that abstained instead of naming the adduct.
#
# STILL excluded (fail-closed): bare metals (organometallic routing owns them,
# never swallowed here -- except at the best-effort tier, see
# ``_METAL_ELEMENT_NAMES`` below) and bare N/ammonia (a bare nitrogen fragment is
# more often a perception artefact than a genuine ammoniate; the exclusion is
# deliberate and left in place until a corpus-grounded reason to add it).
SINGLE_ATOM_COMPONENT_NAMES: Dict[str, str] = {
    "O": "water",
    "F": "hydrogen fluoride",
    "Cl": "hydrogen chloride",
    "Br": "hydrogen bromide",
    "I": "hydrogen iodide",
    "C": "methane",           # -1a: CH4, the dominant nonmetal co-component
    "S": "hydrogen sulfide",  # -1a: H2S
    "P": "phosphane",         # -1a: PH3 (phosphane is the PIN, not phosphine)
}


# Breadth Job 2 (best-effort tier only): the monoatomic metal components of a
# disconnected depiction -- 'CC(N)C(=O)O.CC(N)C(=O)O.[Ni]', six '[C-]#N' with
# '[Fe+2]'. "Mixed organic - inorganic adducts" (the Blue Book):
# "organic components in order as described in, inorganic components in
# order as described in Ref 12"; and (:4667) "preferred IUPAC names cannot be
# assigned to mixed adducts because preferred IUPAC names have not yet been
# determined for inorganic components". (:39735) notes no PIN for the
# Group 1-12 metals either. Such a name is therefore never a PIN: it is built only
# at the best-effort tier, labelled below PIN (namer._is_metal_adduct_without_pin)
# and shipped only when OPSIN reads it back to exactly the drawn structure
# (``_parse_reproduces_depiction`` below, then the full-key round trip).
#
# A metal atom takes its element name; a metal cation adds its charge number in
# parentheses, the notation of 'pentaammine(ethanido)osmium(1+) chloride'
#,:39793). The element set is perception.metals.METAL_ELEMENT_SYMBOLS;
# every spelling below was read back by OPSIN 2.9.0 to the bare atom and to its
# 1+ and 2+ ions (all 94).
_METAL_ELEMENT_NAMES: Dict[str, str] = {
    'Ac': 'actinium', 'Ag': 'silver', 'Al': 'aluminium', 'Am': 'americium',
    'As': 'arsenic', 'Au': 'gold', 'B': 'boron', 'Ba': 'barium',
    'Be': 'beryllium', 'Bh': 'bohrium', 'Bi': 'bismuth', 'Bk': 'berkelium',
    'Ca': 'calcium', 'Cd': 'cadmium', 'Ce': 'cerium', 'Cf': 'californium',
    'Cm': 'curium', 'Cn': 'copernicium', 'Co': 'cobalt', 'Cr': 'chromium',
    'Cs': 'caesium', 'Cu': 'copper', 'Db': 'dubnium', 'Ds': 'darmstadtium',
    'Dy': 'dysprosium', 'Er': 'erbium', 'Es': 'einsteinium', 'Eu': 'europium',
    'Fe': 'iron', 'Fm': 'fermium', 'Fr': 'francium', 'Ga': 'gallium',
    'Gd': 'gadolinium', 'Ge': 'germanium', 'Hf': 'hafnium', 'Hg': 'mercury',
    'Ho': 'holmium', 'Hs': 'hassium', 'In': 'indium', 'Ir': 'iridium',
    'K': 'potassium', 'La': 'lanthanum', 'Li': 'lithium', 'Lr': 'lawrencium',
    'Lu': 'lutetium', 'Md': 'mendelevium', 'Mg': 'magnesium', 'Mn': 'manganese',
    'Mo': 'molybdenum', 'Mt': 'meitnerium', 'Na': 'sodium', 'Nb': 'niobium',
    'Nd': 'neodymium', 'Ni': 'nickel', 'No': 'nobelium', 'Np': 'neptunium',
    'Os': 'osmium', 'Pa': 'protactinium', 'Pb': 'lead', 'Pd': 'palladium',
    'Pm': 'promethium', 'Po': 'polonium', 'Pr': 'praseodymium', 'Pt': 'platinum',
    'Pu': 'plutonium', 'Ra': 'radium', 'Rb': 'rubidium', 'Re': 'rhenium',
    'Rf': 'rutherfordium', 'Rg': 'roentgenium', 'Rh': 'rhodium', 'Ru': 'ruthenium',
    'Sb': 'antimony', 'Sc': 'scandium', 'Sg': 'seaborgium', 'Si': 'silicon',
    'Sm': 'samarium', 'Sn': 'tin', 'Sr': 'strontium', 'Ta': 'tantalum',
    'Tb': 'terbium', 'Tc': 'technetium', 'Te': 'tellurium', 'Th': 'thorium',
    'Ti': 'titanium', 'Tl': 'thallium', 'Tm': 'thulium', 'U': 'uranium',
    'V': 'vanadium', 'W': 'tungsten', 'Y': 'yttrium', 'Yb': 'ytterbium',
    'Zn': 'zinc', 'Zr': 'zirconium',
}


def _metal_atom_component_name(frag_mol) -> Optional[str]:
    """The name of a one-atom metal component, or None.

    'nickel' for [Ni], 'iron(2+)' for [Fe+2], '(99Tc)technetium' for [99Tc]
    (the nuclide descriptor of in front of the element name). None for
    a non-metal, a metal carrying hydrogen ([NaH] is a hydride, not a metal atom)
    or a negative metal ion (an '-ide' of a metal is not built here).
    """
    if frag_mol.GetNumAtoms() != 1:
        return None
    atom = frag_mol.GetAtomWithIdx(0)
    name = _METAL_ELEMENT_NAMES.get(atom.GetSymbol())
    if name is None or atom.GetTotalNumHs() != 0:
        return None
    charge = atom.GetFormalCharge()
    if charge < 0:
        return None
    if charge > 0:
        name = f"{name}({charge}+)"
    if atom.GetIsotope():
        name = f"({atom.GetIsotope()}{atom.GetSymbol()}){name}"
    return name


# Terminal ligands of a mononuclear metal halide / oxide component, keyed by
# (atomic number, bond order to the metal) -> ligand prefix. The prefixes are the
# ones OPSIN 2.9.0 reads in an additive name ('dichloropalladium' ->
# Cl[Pd]Cl, 'trichlorooxovanadium' -> O=[V](Cl)(Cl)Cl); the 2005 ligand spellings
# of the Blue Book ('chlorido', 'iodido') are not read by it, so a name
# built with them could never pass the round trip.
_METAL_HALIDE_OXIDE_LIGANDS: Dict[Tuple[int, float], str] = {
    (9, 1.0): 'fluoro', (17, 1.0): 'chloro', (35, 1.0): 'bromo',
    (53, 1.0): 'iodo', (8, 2.0): 'oxo',
}

_LIGAND_MULTIPLIERS = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}


def _metal_halide_oxide_component_name(frag_mol) -> Optional[str]:
    """Additive name of a neutral mononuclear metal halide / oxide component, or None.

    'dichloropalladium' for Cl[Pd]Cl (the inorganic component of
    'cycloocta-1,5-diene--dichloropalladium (1/1)'), 'oxovanadium' for O=[V],
    'iodocopper' for [Cu]I: the ligand prefixes in alphanumerical order, each with
    its multiplier, then the element name. Only for a true metal (the Groups 1-12
    metals and the f-block, perception.metals._TRUE_METAL_SYMBOLS_FOR_VETO: a
    Group 13-16 element has a parent hydride and is named substitutively,
    with no charge and no hydrogen, whose every other atom is an uncharged,
    hydrogen-free terminal halogen (single bond) or oxygen (double bond) on it.
    """
    from ..perception.metals import _TRUE_METAL_SYMBOLS_FOR_VETO
    metals = [a for a in frag_mol.GetAtoms() if a.GetSymbol() in _METAL_ELEMENT_NAMES]
    if len(metals) != 1 or frag_mol.GetNumAtoms() < 2:
        return None
    metal = metals[0]
    if (metal.GetSymbol() not in _TRUE_METAL_SYMBOLS_FOR_VETO
            or metal.GetFormalCharge() != 0 or metal.GetTotalNumHs() != 0
            or metal.GetIsotope()):
        return None
    counts: Dict[str, int] = {}
    for atom in frag_mol.GetAtoms():
        if atom.GetIdx() == metal.GetIdx():
            continue
        if (atom.GetFormalCharge() != 0 or atom.GetTotalNumHs() != 0
                or atom.GetIsotope() or atom.GetDegree() != 1
                or atom.GetNumRadicalElectrons() != 0):
            return None
        bond = frag_mol.GetBondBetweenAtoms(atom.GetIdx(), metal.GetIdx())
        if bond is None:
            return None
        prefix = _METAL_HALIDE_OXIDE_LIGANDS.get(
            (atom.GetAtomicNum(), bond.GetBondTypeAsDouble()))
        if prefix is None:
            return None
        counts[prefix] = counts.get(prefix, 0) + 1
    if any(n not in _LIGAND_MULTIPLIERS for n in counts.values()):
        return None
    ligands = "".join(_LIGAND_MULTIPLIERS[counts[p]] + p for p in sorted(counts))
    return ligands + _METAL_ELEMENT_NAMES[metal.GetSymbol()]


def _is_metal_atom_fragment(frag_mol) -> bool:
    """True iff the fragment is one metal atom (any charge, any hydrogen count)."""
    return (frag_mol.GetNumHeavyAtoms() == 1
            and any(a.GetSymbol() in _METAL_ELEMENT_NAMES
                    for a in frag_mol.GetAtoms()))


def _parse_reproduces_depiction(name: str, mol) -> bool:
    """True iff OPSIN reads ``name`` back to exactly the drawn structure.

    Equal canonical isomeric SMILES: the same fragments with the same atoms,
    bonds, formal charges, hydrogen counts, isotopes and stereo. This is stricter
    than an equal standard InChIKey, which does not see where a charge or a
    mobile hydrogen sits (a zwitterion and its neutral form share one key), and it
    is what a name for a disconnected ionic depiction has to prove: OPSIN balances
    the charges of an adduct's components itself ('triethylphosphanium--copper
    monoiodide (1/1)' reads back as [Cu+]I). A name OPSIN cannot read, or an
    unavailable OPSIN, fails closed.
    """
    from orthonym.namer import _same_canonical_smiles, _validity_gate_name_to_smiles
    try:
        parsed = _validity_gate_name_to_smiles(name)
        return bool(parsed) and _same_canonical_smiles(parsed, Chem.MolToSmiles(mol))
    except Exception:
        return False


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


def _name_component(frag_smi: str, style: str, *,
                    general_fallback: bool = False,
                    allow_aromatic_general: bool = False,
                    general_fallback_unverified: bool = False,
                    charged_ok: bool = False,
                    ions_ok: bool = False) -> Optional[str]:
    """Name ONE component fragment, or None (fail-closed).

    Single-heavy-atom fragments come ONLY from the table above.
    Multi-atom fragments go through the full single-component pipeline via
    a FRESH Orthonym instance (the audit fresh-instance pattern,
    same as routing/dispatch_table._handle_multi_component_neutral), so the
    per-fragment OPSIN validity gate stays ON in production.

    : ``general_fallback`` / ``allow_aromatic_general`` (default False ->
    byte-identical PIN behaviour) select the ``complete`` tier for the
    per-component namer, so a component nameable only by the general engine
    (e.g. a silyl-heteroarene, a von-Baeyer polyene cage) is named rather than
    dropping the whole adduct to a fail-closed abstention. Charge/single-atom
    scope is unchanged: a charged fragment still refuses here (P5 owns charged).
    """
    frag_mol = Chem.MolFromSmiles(frag_smi)
    if frag_mol is None:
        return None
    is_charged = Chem.GetFormalCharge(frag_mol) != 0
    if is_charged and not charged_ok:
        return None  # charged fragments belong to the salt/ion router
    # -1d: a multi-atom charged ion (charged_ok) is named as a substitutive
    # ion word (…-ium / …-ide) by the fresh best-effort instance below, exactly as
    # a neutral multi-atom fragment. A single charged atom (a bare ion) stays out
    # of the single-atom table and refuses.
    #
    # Breadth Job 2 (``ions_ok``, best-effort tier only): a one-atom metal
    # component takes its element name ('nickel', 'iron(2+)'), and a one-atom
    # charged non-metal ('[NH2-]' -> 'azanide') is named by the fresh instance
    # below like any other ion.
    if frag_mol.GetNumHeavyAtoms() == 1:
        if ions_ok and _is_metal_atom_fragment(frag_mol):
            return _metal_atom_component_name(frag_mol)
        if not (ions_ok and is_charged):
            return None if is_charged else SINGLE_ATOM_COMPONENT_NAMES.get(
                Chem.MolToSmiles(frag_mol, canonical=True))
    from orthonym.namer import Orthonym  # lazy: avoid import cycle
    if ions_ok and _metal_halide_oxide_component_name(frag_mol) is not None:
        # Breadth Job 2: a metal halide / oxide the single-component pipeline
        # has no name for (it abstains on Cl[Pd]Cl, O=[V], [Cu]I).
        return _metal_halide_oxide_component_name(frag_mol)
    try:
        name = Orthonym(
            style=style, general_fallback=general_fallback,
            allow_aromatic_general=allow_aromatic_general,
            general_fallback_unverified=general_fallback_unverified).name(frag_smi)
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
    # unparseable component in EVERY context. 'unavailable' (no JAR / transient
    # OPSIN error) is not a rejection, so the component stays; the production
    # gate then checks the whole adduct name and fails CLOSED on 'unavailable'
    # (TRIAGE g7 C01).
    #
    # Claims conformance part 2 (2026-09-27): a component OPSIN rejects is refused
    # unless it is an exact-match list name (namer._is_exact_match_list_name), the
    # same rule as for a whole shown name. 594f8a788 had let a component with a
    # pseudoasymmetric (lowercase r/s) descriptor through when the stripped-form
    # round trip plus the centres labeller agreed ('(1R,3r,5S)-tropan-3-yl
    # 1H-indole-3-carboxylate' for tropisetron); OPSIN 2.9.0 reads no r/s, so the
    # salt name built on it could not be read back either, and at the default
    # tier it shipped on the stereo-stripped branch.
    from orthonym.namer import _is_exact_match_list_name, _validity_gate_status
    if (_validity_gate_status(name) == "rejected"
            and not _is_exact_match_list_name(frag_smi, name)):
        return None
    return name


def _component_bucket(frag_mol, frag_smi: str) -> int:
    """/ citation buckets: 0 organic, 1 inorganic, 2 water."""
    if frag_smi == "O":
        return 2  # "water (if present), is cited last"
    if any(a.GetAtomicNum() == 6 for a in frag_mol.GetAtoms()):
        return 0  # "organic compounds precede inorganic compounds"
    return 1


def component_sort_key(frag_smi: str) -> Tuple[int, int, int, str]:
    """Deterministic citation order for one component.

    (bucket, seniority index, -heavy_atoms, canonical_smiles):
    organic components ordered by the seniority of the class of their
    principal characteristic group: "cited in the order of
    seniority of classes (see "); components without a suffix-capable
    PCG (hydrocarbons, N-heterocycles-as-π-bases) rank after all
    PCG-bearing organics; ties by descending size then canonical SMILES —
    reproduces every Blue Book example (coronene—trinitrobenzene
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


#: "an appropriate numerical prefix, such as 'mono','di', 'tri'"
_HYDRATE_MULTIPLIERS = {
    1: "mono", 2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}


def _hydrate_word_form(named: List[Tuple[str, int]],
                       water_index: int) -> Optional[str]:
    """General-nomenclature '<components> <prefix>hydrate', or None.

    Defined ONLY when every non-water component shares one count p; the
    prefix encodes the reduced water:parent ratio w/p — n/1 -> mono/di/
    tri/..., 1/2 -> hemi, 3/2 -> sesqui line 4657; the Blue Book
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
                style: str = "pin", *,
                general_fallback: bool = False,
                allow_aromatic_general: bool = False,
                general_fallback_unverified: bool = False) -> Optional[str]:
    """Name an all-neutral multi-component input per, or None.

    Fail-closed refusals (return None; the dispatch cascade then falls
    through to the honest 'unknown organic compound'):
      * fewer than 2 DISTINCT components (identical-only sets are not
        adducts — the dispatch handler keeps the frozen space-join there);
      * no multi-atom component at all;
      * any single-heavy-atom fragment outside SINGLE_ATOM_COMPONENT_NAMES
        (bare metals -> organometallic routing, never swallowed here);
      * any charged fragment (salt/ion routing owns charged input);
      * ANY component the single-component pipeline cannot name.

    : ``general_fallback`` / ``allow_aromatic_general`` (default False ->
    byte-identical PIN output) select the ``complete`` tier for the
    per-component namer (see:func:`_name_component`), so a multi-fragment
    input whose only unnameable part was a general-engine-only component
    (silyl-heteroarene, von-Baeyer polyene cage,...) is named under
    ``complete`` instead of abstaining. All other scope (charge / single-atom /
    proportion assembly / ordering) is unchanged.
    """
    components = split_components(mol)
    if components is None or len(components) < 2:
        return None
    frag_mols = {smi: Chem.MolFromSmiles(smi) for smi, _ in components}
    if any(fm is None for fm in frag_mols.values()):
        return None
    # Breadth Job 2: the metal-adduct widening runs at the best-effort tier only
    #, the Blue Book: no PIN for a mixed organic-inorganic adduct),
    # and only for an assembly with a metal atom in it; a metal-free assembly is
    # named exactly as before.
    _ions_ok = bool(general_fallback_unverified) and any(
        a.GetSymbol() in _METAL_ELEMENT_NAMES for a in mol.GetAtoms())
    _has_metal_atom = _ions_ok and any(
        _is_metal_atom_fragment(fm) for fm in frag_mols.values())
    if not _has_metal_atom and not any(
            fm.GetNumHeavyAtoms() >= 2 for fm in frag_mols.values()):
        return None
    if all(_is_metal_atom_fragment(fm) for fm in frag_mols.values()):
        return None  # metal atoms only: not a mixed adduct
    # -1d: under best-effort, a net-charged multi-fragment assembly composes
    # as a -notation adduct of its (charged) ion components -- 'cation—anion
    # (1/1)' (measured 65% RT_FULL, 0 wrong). OPSIN preserves the net charge when a
    # cation IS present, but it PROTONATES a lone anion (e.g. 'acetate—water (1/1)'
    # -> neutral acetic acid), so 0-wrong is delivered by the full-InChIKey
    # /general-fallback gate (namer.py), NOT by OPSIN or this producer -- the
    # charge-mismatch cases are emitted here and SUPPRESSED downstream. PIN tier
    # keeps the charge refusal. Out-of-scope metal assemblies are EXCLUDED (0-wrong:
    # never render a coordination complex / organometallic).
    _best_effort = (general_fallback or general_fallback_unverified
                    or allow_aromatic_general)
    from ..perception.metals import assembly_has_out_of_scope_metal
    _out_of_scope_metal = assembly_has_out_of_scope_metal(mol)
    # Breadth Job 2: at the best-effort tier the charged components of a metal
    # assembly are named too; the name then has to reproduce the drawn charges
    # exactly (``_parse_reproduces_depiction`` below).
    _charged_ok = _best_effort and (_ions_ok or not _out_of_scope_metal)
    _widened = False
    for smi, fm in frag_mols.items():
        _charged = Chem.GetFormalCharge(fm) != 0
        if _charged and not _charged_ok:
            return None
        if _charged and _out_of_scope_metal:
            _widened = True
        if (fm.GetNumHeavyAtoms() == 1
                and smi not in SINGLE_ATOM_COMPONENT_NAMES):
            if not (_ions_ok and (_charged or _is_metal_atom_fragment(fm))):
                return None
            _widened = True
        if _ions_ok and _metal_halide_oxide_component_name(fm) is not None:
            _widened = True
    ordered = sorted(components, key=lambda t: component_sort_key(t[0]))
    named: List[Tuple[str, int]] = []
    for smi, count in ordered:
        component_name = _name_component(
            smi, style, general_fallback=general_fallback,
            allow_aromatic_general=allow_aromatic_general,
            general_fallback_unverified=general_fallback_unverified,
            charged_ok=_charged_ok, ions_ok=_ions_ok)
        if component_name is None:
            return None  # fail-closed: never drop or placeholder a component
        named.append((component_name, count))
    if _widened:
        _name = _assemble_adduct_name(named)
        return _name if _parse_reproduces_depiction(_name, mol) else None
    if style == "general":
        water_indices = [i for i, (smi, _c) in enumerate(ordered)
                         if smi == "O"]
        if water_indices:
            word = _hydrate_word_form(named, water_indices[0])
            if word is not None:
                return word
    return _assemble_adduct_name(named)
