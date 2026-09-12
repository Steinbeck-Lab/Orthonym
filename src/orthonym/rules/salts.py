"""
Salt and zwitterion naming rules per IUPAC 2013.

Handles naming of:
- Salts: Compositional nomenclature (cation + anion as separate words)
- Zwitterions: Internal ion pairs with combined suffixes

IUPAC 2013 References:
-: Anion nomenclature
-: Cation nomenclature
-: Zwitterion nomenclature

Key naming patterns:
- Salts: "cation anion" format (sodium acetate, ammonium chloride)
- Zwitterions: base name with ionic suffixes (2-azaniumylacetate)
- Multiple ions: alphabetized cations before alphabetized anions
- Stoichiometry: multiplicative prefixes for repeated ions (diacetate)
"""

import itertools
from collections import Counter
from typing import Any, Dict, List, Optional

from rdkit import Chem

from ..data.ion_retained_names import INORGANIC_ANIONS, INORGANIC_CATIONS
from ..errors import is_failure_name
from ..perception.ions import get_ion_sites, parse_salt_fragments
from ..perception.metals import assembly_has_out_of_scope_metal
from .ions import name_anion, name_cation
from ..perception.smarts_cache import compiled as _compiled_smarts

# B2: basic (protonatable) amine nitrogen -- an sp3 N that is not an amide,
# sulfonamide, N-oxide/N-N/N-halide, nitro, or already-charged/aromatic N. Used
# by the [H+]-diamine-hemisalt transform to place a floating proton.
_BASIC_AMINE_N = Chem.MolFromSmarts(
    "[NX3;!$([NX3]=*);!$([NX3][CX3]=[OX1,SX1]);!$([NX3][SX4]);"
    "!$([NX3][#7,#8]);!$([N+]);!$([n])]")


# === VARIABLE-VALENCE METALS (charged-species fix, 169.6 caveat /) ===
# Metals that exhibit more than one common oxidation state and therefore carry a
# Stock oxidation-state numeral in their salt cation word (.4.2.2 /:
# e.g. gold(I) chloride, iron(II/III). FIXED-valence metals (group 1/2, Al, Zn, Ag,
# Sc, Ge,...) do NOT carry a Stock numeral (sodium chloride, calcium dichloride).
# This restores the 169.6-pre 'gold(I) chloride' that the salt path regressed to
# 'gold chloride' (audit Dim-08 §B Cause 2, with the framing correction).
_VARIABLE_VALENCE_METALS = frozenset({
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",          # 3d transition (variable)
    "Mo", "W", "Tc", "Re", "Ru", "Os", "Rh", "Ir", "Pd", "Pt",  # 4d/5d transition
    "Au", "Hg", "Sn", "Pb", "Tl", "Sb", "Bi", "Ce", "Eu", "Sm", "Yb", "U",
})


def _with_stock_if_variable_valence(frag_mol, word: str) -> str:
    """Append the Stock oxidation-state numeral to a salt cation word IFF the cation
    is a MONATOMIC variable-valence metal (.4.2.2). For a monatomic metal cation
    the oxidation state equals the formal charge. Fixed-valence metals are unchanged."""
    if frag_mol.GetNumHeavyAtoms() != 1:
        return word
    atom = frag_mol.GetAtomWithIdx(0)
    if atom.GetSymbol() not in _VARIABLE_VALENCE_METALS:
        return word
    ox = atom.GetFormalCharge()
    if ox <= 0:
        return word
    try:
        from .organometallics import _to_roman
        return f"{word}({_to_roman(ox)})"
    except (ValueError, ImportError):
        return word


# === STOICHIOMETRIC PREFIXES ===

STOICHIOMETRIC_PREFIXES = {
    2: 'di',
    3: 'tri',
    4: 'tetra',
    5: 'penta',
    6: 'hexa',
    7: 'hepta',
    8: 'octa',
    9: 'nona',
    10: 'deca',
}


# (:41013) / (:41431): a COMPOSITE ion is multiplied
# with the enclosing multipliers bis/tris/tetrakis (name wrapped in parens),
# NOT the simple di/tri/tetra (which are glued directly onto the name).
COMPLEX_STOICHIOMETRIC_PREFIXES: Dict[int, str] = {
    2: 'bis', 3: 'tris', 4: 'tetrakis', 5: 'pentakis', 6: 'hexakis',
    7: 'heptakis', 8: 'octakis', 9: 'nonakis', 10: 'decakis',
}


# / (avoid-ambiguity): a bare mononuclear-oxoanion word W
# whose SIMPLE-multiplied form ``di<W>`` is ITSELF a real OPSIN word for a
# DIFFERENT, condensed poly species (pyro/di-nuclear). For those the ``di``/``tri``
# multiplier collides — ``diphosphate`` = P2O7 (pyrophosphate), not 2×PO4;
# ``disulfate`` = S2O7; ``dicarbonate`` = C2O5; ``dihydrogensulfate`` = neutral
# H2SO4 — so a count≥2 salt of W MUST take the enclosing multiplier bis(W)/tris(W).
# Emitting ``di<W>`` names the wrong molecule (then suppresses it, so the
# salt needlessly ABSTAINS even though ``bis(W)`` round-trips).
#
# COMPLETE + EXACT-MATCH set — every entry VERIFIED 2026-08-23 by
# internal notes + a temp dir
# probe_di_collision_complete.py: for each W, opsin_parse("di"+W) returns a SINGLE
# connected RDKit fragment (a distinct condensed species). Probed the full
# candidate universe = every INORGANIC_ANIONS value + every single-word oxoanion
# the salt anion-naming path can emit + controls; controls
# (acetate/chloride/bromide/benzoate/methanesulfonate/…) did NOT collide and are
# absent. ``azanide``/``phosphonate`` also collide as bare words but are covered —
# together with their SUBSTITUTED parents (methylphosphonate, …) — by the retained
# ``endswith`` clause below, so they are deliberately not duplicated here.
# Enclosing forms (bis/tris) RT-verified for every reachable member; the only
# currently-passing count≥2 rows (Al2(SO4)3 → trisulfate, hexasodium trisulfite)
# move to tris(sulfate)/tris(sulfite), both RT-valid → 0 regressions.
_DI_COLLISION_ANIONS = frozenset({
    'amidosulfate', 'borate', 'carbonate', 'chromate', 'germanate',
    'hydrogensulfate', 'peroxydisulfate', 'phosphate', 'phosphite', 'selenate',
    'selenide', 'selenite', 'silicate', 'sulfate', 'sulfite', 'tellurate',
    'tellurite', 'thiosulfate', 'thiosulfite',
})


def _ion_needs_enclosing_multiplier(name: str) -> bool:
    """True if an ion name must take bis(...)/tris(...) rather than di.../tri...

    Applied to BOTH anion and cation words (called from
    _apply_stoichiometric_prefix, which formats both halves of a salt) —
    hence "ion", not "anion", in the name.

    Orthonym's 0-wrong gate IS OPSIN round-trip, so the trigger set matches
    OPSIN 2.9.0's grammar (probe_salt_spelling.py, 2026-08-23) — STRICTER than
    a naive multiplier predicate that never validates against
    OPSIN and would emit simple `di` for `2-hydroxypropanoate` (OPSIN rejects
    `di2-hydroxypropanoate`). This is a dedicated salt predicate on purpose;
    do NOT reuse the substituent-tuned is_complex_substituent/get_multiplier_prefix
    (see the ⛔ warning in get_multiplier_prefix's docstring, commit b3f6ce7c).

    Composite when the ion name:
      - is a bare mononuclear-oxoanion word whose ``di<name>`` collides with a
        real condensed poly species (``_DI_COLLISION_ANIONS``, e.g. phosphate →
        diphosphate=P2O7), or
      - contains '(', ')' or a space (already enclosed / multi-word), or
      - contains any digit (a locant; `di<name>` fuses ambiguously), or
      - contains '-' (catches no-digit stereo prefixes: D-/L-), or
      - starts with a multiplier word, or ends with a stem-collision suffix
        where a bare `di-` would fuse into a different word (carried from
        where a bare `di-` would fuse into a different word).
    """
    if name in _DI_COLLISION_ANIONS:
        return True
    if '(' in name or ')' in name or ' ' in name:
        return True
    if any(ch.isdigit() for ch in name):
        return True
    if '-' in name:
        return True
    # BREADTH-UNVERIFIED (Milestone C3, salt breadth program): the startswith
    # multiplier-word check above and the endswith stem-collision check below
    # were not confirmed against an OPSIN round-trip witness at A1 time —
    # revisit with an OPSIN-RT check when C3 reaches this area.
    if name.startswith(('bis', 'tris', 'tetrakis', 'tetra', 'penta', 'hexa')):
        return True
    if name.endswith(('azanide', 'phosphinate', 'phosphonate')):
        return True
    return False


# === AMINO ACID ZWITTERION PATTERNS ===

# SMARTS for alpha-amino acid zwitterion pattern
ALPHA_AA_ZWITTERION = '[NX4+;H3][CX4][CX3](=[OX1])[OX1-]'

# SMARTS for beta-amino acid zwitterion pattern (2-carbon gap)
BETA_AA_ZWITTERION = '[NX4+;H3][CX4][CX4][CX3](=[OX1])[OX1-]'

# SMARTS for gamma-amino acid zwitterion pattern (3-carbon gap)
GAMMA_AA_ZWITTERION = '[NX4+;H3][CX4][CX4][CX4][CX3](=[OX1])[OX1-]'


# === RETAINED AMINO ACID NAMES ===

# Map canonical SMILES of zwitterion form to trivial name.
# Both chirality variants are included where the canonical SMILES
# differs depending on input notation (e.g., @@ vs @).
RETAINED_AMINO_ACID_ZWITTERIONS = {
    # Glycine zwitterion
    '[NH3+]CC(=O)[O-]': 'glycine',
    # Alanine zwitterion
    'C[C@H]([NH3+])C(=O)[O-]': 'L-alanine',
    'C[C@@H]([NH3+])C(=O)[O-]': 'D-alanine',
    # NOTE (charged B1): the achiral (stereo-UNDEFINED) 'CC([NH3+])C(=O)[O-]'
    # entry mapping to bare 'alanine' was DELETED. Per the Blue Book
    # "The stereodescriptors 'D' and 'L'"), a bare retained
    # amino-acid name denotes ONLY the defined (L) configuration -- OPSIN's
    # grammar always resolves 'alanine' to the L stereocentre (verified:
    # opsin_parse('alanine') -> InChIKey QNAYBMKLOCPYGJ-REOHCLBHSA-N), which
    # provably differs from the stereo-undefined input's InChIKey
    # (QNAYBMKLOCPYGJ-UHFFFAOYSA-N) -- a different, more specific claim than
    # the input supports. Same defect class already fixed for the NEUTRAL
    # form (see tests/unit/test_amino_acids.py module docstring); the deleted
    # entry now falls through to `_name_amino_acid_zwitterion`'s existing
    # is_bare_standard_aa systematic-name path (0-wrong: a name that does not
    # over-claim stereochemistry).
    # Valine zwitterion
    'CC(C)[C@@H]([NH3+])C(=O)[O-]': 'D-valine',
    'CC(C)[C@H]([NH3+])C(=O)[O-]': 'L-valine',
    # Leucine zwitterion
    'CC(C)C[C@@H]([NH3+])C(=O)[O-]': 'D-leucine',
    'CC(C)C[C@H]([NH3+])C(=O)[O-]': 'L-leucine',
    # Isoleucine zwitterion
    'CC[C@H](C)[C@@H]([NH3+])C(=O)[O-]': 'D-alloisoleucine',
    'CC[C@H](C)[C@H]([NH3+])C(=O)[O-]': 'L-isoleucine',
    # Serine zwitterion
    '[NH3+][C@@H](CO)C(=O)[O-]': 'L-serine',
    '[NH3+][C@H](CO)C(=O)[O-]': 'D-serine',
    # Threonine zwitterion
    'C[C@@H](O)[C@@H]([NH3+])C(=O)[O-]': 'D-allothreonine',
    'C[C@H](O)[C@@H]([NH3+])C(=O)[O-]': 'D-threonine',
    # Proline zwitterion
    'O=C([O-])[C@@H]1CCC[NH2+]1': 'L-proline',
    # Phenylalanine zwitterion
    '[NH3+][C@@H](Cc1ccccc1)C(=O)[O-]': 'L-phenylalanine',
    # Tyrosine zwitterion
    '[NH3+][C@@H](Cc1ccc(O)cc1)C(=O)[O-]': 'L-tyrosine',
    # Tryptophan zwitterion
    '[NH3+][C@@H](Cc1c[nH]c2ccccc12)C(=O)[O-]': 'L-tryptophan',
    # Methionine zwitterion
    'CSCC[C@@H]([NH3+])C(=O)[O-]': 'D-methionine',
    # Histidine zwitterion
    '[NH3+][C@@H](Cc1c[nH]cn1)C(=O)[O-]': 'L-histidine',
    # Glutamic acid zwitterion (one COOH protonated)
    '[NH3+][C@@H](CCC(=O)O)C(=O)[O-]': 'L-glutamic acid',
    # Aspartic acid zwitterion (one COOH protonated)
    '[NH3+][C@@H](CC(=O)O)C(=O)[O-]': 'L-aspartic acid',
    # Beta-alanine zwitterion (beta-amino acid)
    '[NH3+]CCC(=O)[O-]': 'beta-alanine',
    # GABA zwitterion (gamma-aminobutyric acid)
    '[NH3+]CCCC(=O)[O-]': '4-aminobutanoic acid',
    # NOTE (169.6-04): the hardcoded betaine literal entry was DELETED. 'betaine'
    # is NOT OPSIN-parseable (the validity gate suppressed it to 'unknown organic
    # compound'); the route_charged GUARD-4 structured producer now emits the
    # RT-correct (trimethylazaniumyl)acetate. No per-molecule literal.
    # L-Carnitine zwitterion
    'C[N+](C)(C)C[C@H](O)CC(=O)[O-]': 'L-carnitine',
    # DL-Carnitine (racemic)
    'C[N+](C)(C)CC(O)CC(=O)[O-]': 'carnitine',
}


# === SALT NAMING ===

# Inorganic acid anion to hydroacid salt name mapping
_HYDROACID_SALT_NAMES = {
    '[Cl-]': 'hydrochloride',
    '[Br-]': 'hydrobromide',
    '[I-]': 'hydroiodide',
    '[F-]': 'hydrofluoride',
}

# Hydrogen prefix multipliers for partial salts
_HYDROGEN_PREFIXES = {
    1: 'hydrogen',
    2: 'dihydrogen',
    3: 'trihydrogen',
}


def _count_protonated_acid_sites(frag_mol) -> int:
    """Count the number of still-protonated acid -OH sites in a partially
    deprotonated anion — used to insert the ``hydrogen`` / ``dihydrogen`` word in
    an acid-salt name (IUPAC /.

    Counts protonated carboxylic acid groups (-COOH) PLUS residual oxoacid -OH on
    a P/S/Se acid centre (a P/S/Se bearing at least one ``=O``) — the latter for
    the di-/polynuclear noncarbon-oxoacid acid salts (``disodium dihydrogen
    diphosphate``). Deprotonated ``-O(-)`` sites are NOT counted.
    """
    from rdkit.Chem import MolFromSmarts
    count = 0
    acid_pat = MolFromSmarts('[CX3](=O)[OX2H1]')
    if acid_pat:
        count += len(frag_mol.GetSubstructMatches(acid_pat))
    # Residual oxoacid -OH on a P/S/Se centre carrying a =O.
    oxoacid_oh = MolFromSmarts('[OX2H1][#15,#16,#34]=O')
    if oxoacid_oh:
        seen = set()
        for m in frag_mol.GetSubstructMatches(oxoacid_oh):
            seen.add(m[0])                           # the -OH oxygen
        count += len(seen)
    return count


def normalize_imbalanced_acid_salt(smiles: str) -> Optional[str]:
    """: normalize a charge-imbalanced acid-salt NOTATION to its
    chemically-valid balanced salt.

    A metal cation + a NEUTRAL polybasic INORGANIC oxoacid, written without the
    balancing deprotonation ([Na+].OC(=O)O = NaHCO3, net +1), is a valid acid salt
    in a malformed charge representation. Deprotonate the acid by the net positive
    charge so the whole pipeline sees the balanced salt (species-type -> 'salt',
    name_salt -> 'sodium hydrogen carbonate', and the self-consistency gate compares
    a balanced net-0 structure). Return the balanced-salt canonical SMILES, or None
    (fail-closed -> caller keeps the original smiles) for any other shape.

    Narrow by construction: fires only when the mol is multi-fragment with net
    charge > 0, exactly one neutral fragment that IS a recognized inorganic oxoacid,
    at least one cation fragment, NO anion fragment, and the acid has enough -OH
    protons to balance the charge — so no organic acid / real salt / plain ion is
    touched.
    """
    from rdkit.Chem import MolFromSmarts

    from .inorganic_acids import name_inorganic_acid

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if Chem.GetFormalCharge(mol) <= 0:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) < 2:
        return None
    cations = [f for f in frags if Chem.GetFormalCharge(f) > 0]
    anions = [f for f in frags if Chem.GetFormalCharge(f) < 0]
    neutrals = [f for f in frags if Chem.GetFormalCharge(f) == 0]
    if anions or not cations or len(neutrals) != 1:
        return None
    acid = neutrals[0]
    if name_inorganic_acid(acid) is None:
        return None

    n_protons = sum(Chem.GetFormalCharge(f) for f in cations)
    oh_pat = MolFromSmarts('[OX2H1]')
    if oh_pat is None:
        return None
    oh_matches = acid.GetSubstructMatches(oh_pat)
    if len(oh_matches) < n_protons:
        return None
    rw = Chem.RWMol(acid)
    for (o_idx,) in oh_matches[:n_protons]:
        o = rw.GetAtomWithIdx(o_idx)
        o.SetFormalCharge(-1)
        o.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    anion_smi = Chem.MolToSmiles(rw)
    parts = [Chem.MolToSmiles(c) for c in cations] + [anion_smi]
    balanced = '.'.join(parts)
    if Chem.MolFromSmiles(balanced) is None:
        return None
    return balanced


def _protonate_amines_with_h_plus(mol, n_protons: int):
    """ B2: consume ``n_protons`` bare [H+] fragments by protonating that many
    basic amine nitrogens, yielding EVERY reconstructed (charge-balanced) mol
    that results from a distinct choice of which amine site(s) to protonate.

    a review #1 (determinism fix): ``GetSubstructMatches`` returns matches in
    ATOM-INDEX order, which is a PARSE-order artifact -- two SMILES spellings
    of the identical canonical molecule can enumerate the same amine sites in
    different orders, so a single ``[:n_protons]`` slice used to make the
    choice of WHICH amine gets protonated spelling-dependent (two
    identical-canonical-SMILES inputs producing different names, or one
    naming and the spelling-equivalent other abstaining). Fixed two ways:

    (a) Candidate sites are sorted by RDKit CANONICAL ATOM RANK
        (``CanonicalRankAtoms``), not raw atom index. The canonical rank is a
        function of the molecular graph alone, not of input atom order, so
        the sort order -- and therefore which site is tried FIRST -- is
        identical across every spelling of the same molecule.
    (b) This now returns/yields ALL ``n_protons``-sized combinations of
        candidate sites (in that same canonical order), not just the first.
        The caller (``name_salt``) tries them in order and BACKTRACKS to the
        next candidate whenever one fails to produce a nameable+round-
        tripping salt, instead of giving up after the first attempt. Every
        candidate still goes through the full ``name_salt`` recursion (every
        existing fail-closed guard applies) and the top-level OPSIN gate, so
        a mis-placed proton still fails closed (0-wrong preserved) -- this
        only widens the set of placements that get a chance, deterministically.

    Every [H+] MUST be consumed (a leftover proton would leave an unbalanced,
    wrongly-charged species), so this yields nothing when there are fewer
    basic amine sites than protons -- the caller then falls through to the
    fail-closed guard.
    """
    try:
        probe = Chem.RWMol(mol)
        amine_idxs = [m[0] for m in probe.GetSubstructMatches(_BASIC_AMINE_N)]
        h_plus_idxs = [a.GetIdx() for a in probe.GetAtoms()
                       if a.GetAtomicNum() == 1 and a.GetFormalCharge() == 1
                       and a.GetDegree() == 0]
        if len(amine_idxs) < n_protons or len(h_plus_idxs) < n_protons:
            return
        # Canonical-rank order: spelling-invariant (depends on the graph, not
        # on input atom order), so the SAME site(s) are tried first for every
        # spelling of the same molecule.
        ranks = list(Chem.CanonicalRankAtoms(probe, breakTies=True))
        amine_idxs = sorted(amine_idxs, key=lambda i: ranks[i])
        h_plus_idxs = sorted(h_plus_idxs, key=lambda i: ranks[i])[:n_protons]
    except (RuntimeError, ValueError):
        return

    for combo in itertools.combinations(amine_idxs, n_protons):
        try:
            rw = Chem.RWMol(mol)
            for n_idx in combo:
                a = rw.GetAtomWithIdx(n_idx)
                a.SetFormalCharge(1)
                a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
                a.SetNoImplicit(True)
            for h_idx in sorted(h_plus_idxs, reverse=True):
                rw.RemoveAtom(h_idx)
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            yield out
        except (RuntimeError, ValueError):
            continue


def _reattach_protons_to_acids(mol):
    """ (Sub-gap 2): reconstruct the NEUTRAL free acid of an organic oxoanion
    written ionically with bare [H+] protons and no neutral base
    (``dicarboxylate.[H+].[H+]``). Give every deprotonated acid oxygen (``[O-]``)
    its proton back and drop every bare ``[H+]``, yielding the single neutral parent
    acid — or None if the shape is not an all-oxygen anion or does not reconstruct
    cleanly to one organic fragment.

    The placement is FORCED, not searched: a salt is charge-balanced, every ``[H+]``
    is +1 and every acid-oxygen site is -1, so ``#[H+] == #[O-]`` and each proton
    returns to one oxygen. There is no site to choose, so the result is deterministic
    and spelling-invariant. The reconstruction is only a CANDIDATE: the caller names
    it and the top-level validity gate RT-verifies the reconstructed parent
    against the ionic input, so a wrong reconstruction fails closed (0-wrong preserved).
    The oxygen filter admits ANY ``[O-]`` (carboxylate/sulfonate/phosphate, but also an
    alkoxide/phenoxide/enolate), NOT only genuine acid oxygens — the safety is the GATE
    CASCADE, not a guaranteed InChIKey collision: for a true oxoanion the neutral acid
    shares the ionic input's full key and the gate passes; for a bare alkoxide
    (``CC[O-].[H+]`` -> ethanol) the reconstructed neutral does NOT share the key
    (whole skeleton block differs), so the gate returns mismatch -> abstain rather than
    ship. Scoped to keep the halide 0-wrong guard (``[H+].[Cl-]``) intact: every anionic
    site must be OXYGEN (a ``[Cl-]``/``[S-]`` anion returns None), and the neutral parent
    must be a SINGLE, carbon-bearing (organic) fragment.
    """
    try:
        rw = Chem.RWMol(mol)
        h_plus_idxs = [a.GetIdx() for a in rw.GetAtoms()
                       if a.GetAtomicNum() == 1 and a.GetFormalCharge() == 1
                       and a.GetDegree() == 0]
        neg_atoms = [a for a in rw.GetAtoms() if a.GetFormalCharge() < 0]
        if not h_plus_idxs or not neg_atoms:
            return None
        # Every anionic site must be OXYGEN (carboxylate / sulfonate / phosphate
        # oxygen). This is what keeps the branch off the halide 0-wrong guard.
        if any(a.GetAtomicNum() != 8 for a in neg_atoms):
            return None
        # The protons must balance the anionic charge EXACTLY (a leftover proton or
        # an unprotonated site would leave a wrongly-charged species).
        neg_total = -sum(a.GetFormalCharge() for a in neg_atoms)
        if neg_total != len(h_plus_idxs):
            return None
        for a in neg_atoms:
            charge = a.GetFormalCharge()
            a.SetFormalCharge(0)
            a.SetNumExplicitHs(a.GetNumExplicitHs() + abs(charge))
        for h_idx in sorted(h_plus_idxs, reverse=True):
            rw.RemoveAtom(h_idx)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
    except (RuntimeError, ValueError):
        return None
    # A single, neutral, carbon-bearing (organic) parent acid -- never a mixture
    # of free acids and never an inorganic oxoacid (out of this sub-gap's scope).
    if Chem.GetFormalCharge(out) != 0:
        return None
    if len(Chem.GetMolFrags(out)) != 1:
        return None
    if not any(a.GetAtomicNum() == 6 for a in out.GetAtoms()):
        return None
    return out


# -1b: alkali + alkaline-earth (Groups 1-2). A net-0 salt of one of these
# with an organic ion is a binary salt (in scope, name_salt's domain). Any
# OTHER true metal is out of scope by default coordination complex), and any
# true metal co-present with a carbanion/hydride is organometallic. The
# census OUT predicate (internal notes _review_census5.py) — the 1b
# widening is scoped to exactly the in-scope set.
def _name_organic_ion_best_effort(ion_smiles: str, *,
                                  style: str = 'general',
                                  general_fallback: bool = False,
                                  general_fallback_unverified: bool = False,
                                  allow_aromatic_general: bool = False) -> str:
    """-1b: name a single organic ion fragment (a cation or anion the
    ordinary ``name_cation`` / ``route_charged`` / ``name_anion`` path could not
    name) via a fresh best-effort ``Orthonym`` instance -- the ``adducts.py``
    per-component pattern. Returns the ion word (``...-ium`` / ``...-ide`` /
    ``...-oate``) or ``''`` on failure.

    Cycle-safe: ``ion_smiles`` is a SINGLE charged fragment (no ``.``), so
    ``is_salt`` (which requires BOTH a cation and an anion fragment) is False
    and the fresh instance never re-enters ``name_salt``.

    Gated by the caller: the widening fires ONLY when a best-effort flag is set,
    so the PIN tier (all flags False) is unchanged. On the PIN path this helper
    is never reached (the caller guards on ``_best_effort``).
    """
    if not (general_fallback or general_fallback_unverified
            or allow_aromatic_general):
        return ''
    try:
        from orthonym.namer import Orthonym
        name = Orthonym(
            style=style,
            general_fallback=general_fallback,
            general_fallback_unverified=general_fallback_unverified,
            allow_aromatic_general=allow_aromatic_general,
        ).name(ion_smiles)
    except Exception:  # never hard-fail a name (matches adducts._name_component)
        return ''
    if not name or is_failure_name(name):
        return ''
    return name


def _ion_fragment_roundtrips(frag_mol, name: str) -> bool:
    """True iff *name* OPSIN-round-trips to *frag_mol*'s FULL isomeric InChIKey.

    Per-component 0-wrong oracle for the salt composer: a cation/anion word is
    trustworthy only if OPSIN parses it back to the exact charged fragment. Used
    to detect a CONFIDENTLY-WRONG ordinary ion name (measured: a fused
    azatetracyclo cage cation named ``tropan-1-ium`` by ``name_cation``, a
    different, smaller ring system) so the best-effort per-component namer can
    replace it instead of being shadowed by the wrong name. Fail CLOSED (no JVM
    / unparseable / any constitution-or-stereo mismatch -> False), so it never
    accepts a wrong ion and never relaxes anything without a working OPSIN.
    """
    if frag_mol is None or not name:
        return False
    try:
        from ..validation.opsin_roundtrip import opsin_roundtrip_check
        smiles = Chem.MolToSmiles(frag_mol)  # canonical, isomeric (stereo kept)
        if not smiles:
            return False
        return opsin_roundtrip_check(smiles, name).get("passed") is True
    except Exception:  # fail-closed: a probe bug must never accept a wrong ion
        return False


def name_salt(mol, style: str = 'pin', *,
              general_fallback: bool = False,
              general_fallback_unverified: bool = False,
              allow_aromatic_general: bool = False) -> str:
    """
    Name a salt using compositional nomenclature.

    Format: cation_name + space + anion_name
    Example: "sodium acetate", "ammonium chloride"

    Also handles:
    - Neutral organic fragments with inorganic counter-ions (Drug.HCl pattern)
    - H+ fragments merged with anions for hydroacid salt naming
    - Partial salts with "hydrogen" prefix (sodium hydrogen fumarate)

    For multiple cations/anions, order alphabetically.
    For stoichiometry > 1, use multiplier prefixes.

    Args:
        mol: RDKit Mol object (contains disconnected fragments)
        style: 'pin' for preferred names

    Returns:
        Salt name as "cation anion" (separate words)

    Example:
        >>> mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        >>> name_salt(mol)
        'sodium acetate'
    """
    if mol is None:
        return ''

    # -1b (0-wrong): the best-effort ion widening is scoped to non-metal +
    # alkali/alkaline-earth salts. If the assembly carries an out-of-scope true
    # metal (a bare transition/other metal, or any true metal with a carbanion/
    # hydride = organometallic), clear the best-effort flags so the metal
    # salt keeps today's behavior and never renders as a compositional salt via a
    # best-effort-named organic ion (the platinum(II)-carbanide 0-wrong hazard).
    if ((general_fallback or general_fallback_unverified or allow_aromatic_general)
            and assembly_has_out_of_scope_metal(mol)):
        general_fallback = False
        general_fallback_unverified = False
        allow_aromatic_general = False

    # a phase-a (0-wrong): a salt is, by definition, charge-balanced
    # overall (the cation charges sum to the anion charges sum -- it is a
    # neutral compound). A disconnected-fragment set carrying a NET charge
    # (e.g. ``CC(=O)[O-].[Pd+2]``, net +1: one acetate anion + a bare Pd2+
    # cation) is NOT a neutral salt -- it is an unbalanced ionic assembly / a
    # charged coordination complex (out of the contributor guide's declared scope,
    # organometallics. Naming it with the ordinary "cation anion" salt
    # grammar silently implies balanced stoichiometry the input does not have:
    # measured, this shipped ``palladium(II) acetate``, which OPSIN round-trips
    # to the BALANCED diacetate Pd(OAc)2, not the 1:1 input (a different
    # molecule). Decline structurally here rather than rely on the
    # OPSIN backstop to catch it after the fact (a project rule: that backstop
    # fails OPEN without a JVM). Mirrors the fail-closed guards below
    # (honest '' -> abstain, never a generic literal).
    if Chem.GetFormalCharge(mol) != 0:
        return ''

    frags = parse_salt_fragments(mol)

    # --- Handle H+ fragments: merge with Cl-/Br- for hydroacid salt naming ---
    # H+ is a bare proton fragment (canonical SMILES: '[H+]')
    h_plus_frags = [f for f in frags['cations'] if f['smiles'] == '[H+]']
    other_cation_frags = [f for f in frags['cations'] if f['smiles'] != '[H+]']
    neutrals = frags.get('neutrals', [])

    # Pattern: Organic_neutral.[H+].[Cl-] -> "organic_name hydrochloride"
    # The H+ merges with Cl- to form HCl, and the neutral organic fragment
    # is the main compound being named as a hydrochloride salt.
    if h_plus_frags and not other_cation_frags and neutrals:
        # Check if all anions are simple halide-type
        hydroacid_names = []
        for anion_frag in frags['anions']:
            hydroacid_name = _HYDROACID_SALT_NAMES.get(anion_frag['smiles'])
            if hydroacid_name:
                hydroacid_names.append(hydroacid_name)

        if hydroacid_names and len(hydroacid_names) == len(frags['anions']):
            # All anions are halides -- name as "organic hydrochloride"
            # Pick the largest neutral organic fragment as the main compound
            organic_neutrals = [
                f for f in neutrals if f['mol'].GetNumHeavyAtoms() > 1
            ]
            if organic_neutrals:
                main_frag = max(organic_neutrals,
                                key=lambda f: f['mol'].GetNumHeavyAtoms())
                try:
                    from ..assembly.fragment_naming import name_fragment_recursively
                    organic_name = name_fragment_recursively(main_frag['smiles'])
                    if organic_name:
                        salt_suffix = ' '.join(sorted(hydroacid_names))
                        return f"{organic_name} {salt_suffix}"
                except (RecursionError, ValueError, RuntimeError):
                    pass

    # B2 ([H+]-diamine hemisalt): a base + organic-acid salt is often written
    # with the proton(s) floating as bare [H+] and the base(s) left NEUTRAL, the
    # acid fully deprotonated (witness 9: 2 amine bases + a fumarate DIANION +
    # 2x[H+]). The hydroacid branch above did NOT fire (the anion is not a halide),
    # so the protons are still orphaned. PROTONATE the basic amine site(s) --
    # consuming one [H+] per site, up to the [H+] budget -- and re-enter name_salt
    # on the reconstructed (charge-balanced) salt, which then routes each cation
    # through the ordinary cation loop. The recursive name_salt + top-level gate
    # verify the result, so a mis-protonation fails closed (0-wrong preserved).
    # a review #1: try EVERY candidate protonation-site combination (in a
    # canonical, spelling-invariant order) and BACKTRACK to the next one if
    # the first fails to name -- never give up after a single try, and never
    # let the choice depend on which N happened to appear first in the input
    # SMILES (identical-canonical-SMILES inputs must give identical output).
    if (h_plus_frags and not other_cation_frags and neutrals
            and frags['anions']):
        for protonated in _protonate_amines_with_h_plus(mol, len(h_plus_frags)):
            recur = name_salt(
                protonated, style, general_fallback=general_fallback,
                general_fallback_unverified=general_fallback_unverified,
                allow_aromatic_general=allow_aromatic_general)
            if recur and not is_failure_name(recur):
                return recur

    # (Sub-gap 2): an organic oxoacid written ionically -- the acid fully
    # deprotonated with its proton(s) floating as bare [H+] and NO neutral base to
    # carry them (e.g. a dicarboxylate DIANION + 2x[H+]). The amine branch above
    # needs a neutral base (absent here), so the protons are still orphaned and the
    # guard below would abstain. REATTACH each proton to its acid oxygen,
    # reconstructing the neutral free acid, and name THAT via the ordinary namer.
    # The top-level gate RT-verifies the emitted name's full InChIKey
    # against the ionic input (identical key), so a wrong reconstruction fails
    # closed. Scoped (in _reattach_protons_to_acids) to all-[H+] cations +
    # all-oxygen anions + single organic parent, so it never touches the halide
    # 0-wrong guard just below.
    if (h_plus_frags and not other_cation_frags and not neutrals
            and frags['anions']):
        neutral_acid = _reattach_protons_to_acids(mol)
        if neutral_acid is not None:
            try:
                from ..assembly.fragment_naming import name_fragment_recursively
                acid_smiles = Chem.MolToSmiles(neutral_acid, canonical=True)
                acid_name = name_fragment_recursively(acid_smiles)
                if acid_name and not is_failure_name(acid_name):
                    return acid_name
            except (RecursionError, ValueError, RuntimeError):
                pass

    # 0-wrong (a review-found): a hydroacid written ionically ([H+].[X-], optionally
    # with water) leaves the proton orphaned unless the hydroacid-merge branch above
    # consumed it (which needs an ORGANIC neutral and RETURNS on success). If any
    # [H+] survives here it would be silently dropped and name_salt would emit an
    # anion-only name (e.g. 'chloride'/'chloride monohydrate') -- a WRONG species
    # (net charge -1). Fail closed. Also closes the pre-existing [H+].[Cl-]->'chloride'
    # default-path 0-wrong bug.
    if h_plus_frags:
        return ''

    cation_names = []
    anion_names = []

    # Process cations (excluding H+ fragments already handled above).
    #: the cation word is the element name (metal) or 'ammonium'
    # (NH4+). The CATION_WORDS table (data/cation_words.py) is the single source
    # of truth, reusing namer._METAL_NAMES; fall back to the INORGANIC_CATIONS
    # retained-name table and then to organic cation naming for substituted
    # ammoniums / carbenium counter-cations.
    from ..data.cation_words import get_cation_word
    cation_list = other_cation_frags if h_plus_frags else frags['cations']
    for cation_frag in cation_list:
        frag_mol = cation_frag['mol']
        smiles = cation_frag['smiles']

        word = get_cation_word(frag_mol)
        if word:
            #: variable-valence metal cations carry the Stock oxidation state
            # (gold(I) chloride); fixed-valence metals (Na/K/Ca/...) do not.
            cation_names.append(_with_stock_if_variable_valence(frag_mol, word))
        elif smiles in INORGANIC_CATIONS:
            cation_names.append(INORGANIC_CATIONS[smiles])
        else:
            # Substituted organic cation (e.g. tetramethylammonium) -> name_cation.
            name = name_cation(frag_mol, style)
            _be = (general_fallback or general_fallback_unverified
                   or allow_aromatic_general)
            # abstain-recovery: name_cation can return a CONFIDENTLY-WRONG
            # name for a complex organic cation (measured: a fused
            # azatetracyclo cage cation -> 'tropan-1-ium', a different, smaller
            # ring system). Being non-empty, that wrong name shadowed the
            # best-effort fallback below (its `if not name:` guard never fired),
            # so ONLY the top-level joined-salt gate caught it and the
            # WHOLE salt abstained -- even though the cation names correctly via
            # the full per-component namer. Under a best-effort flag, verify the
            # ordinary cation name round-trips to THIS fragment; if it does not,
            # replace it with the best-effort per-component name (itself
            # RT-verified by the fresh instance's own gate). Replace ONLY when
            # best-effort yields a verified alternative, so a correct ordinary
            # name is never discarded -- PIN tier + canary salts stay
            # byte-identical (PIN passes all flags False, so `_be` is False).
            if _be and name and not _ion_fragment_roundtrips(frag_mol, name):
                alt = _name_organic_ion_best_effort(
                    smiles, style=style, general_fallback=general_fallback,
                    general_fallback_unverified=general_fallback_unverified,
                    allow_aromatic_general=allow_aromatic_general)
                if alt:
                    name = alt
            if not name:
                # -1b: best-effort fallback for a complex organic cation the
                # ordinary namer cannot build (e.g. a fused-ring azabicyclo-ium).
                # Best-effort-gated -> PIN tier unchanged.
                name = _name_organic_ion_best_effort(
                    smiles, style=style, general_fallback=general_fallback,
                    general_fallback_unverified=general_fallback_unverified,
                    allow_aromatic_general=allow_aromatic_general)
            if name:
                cation_names.append(name)
            # Skip unnamed cations rather than using generic 'cation'

    # fail-closed (0-wrong): every cation fragment IN SCOPE for this loop
    # must have produced a name. `cation_list` already excludes H+ fragments
    # that were merged/attempted above (:287-330), so this cannot fire on the
    # legitimate hydroacid-salt H+ merge -- only on a cation this loop itself
    # could not name (abort-whole:
    # name every fragment or emit nothing).
    if len(cation_names) != len(cation_list):
        return ''

    # Process anions. /: name the ORGANIC anion via the
    # route_charged chokepoint (it owns the parent decision: -oate / -olate /
    # -sulfonate / -ide), falling back to name_anion and the INORGANIC_ANIONS
    # retained table (chloride / sulfate / phosphate — which route_charged does
    # not name). The cation is NOT substituted in (salt = functionalization).
    from .charged_router import route_charged
    for anion_frag in frags['anions']:
        frag_mol = anion_frag['mol']
        smiles = anion_frag['smiles']

        if smiles in INORGANIC_ANIONS:
            anion_names.append(INORGANIC_ANIONS[smiles])
            continue
        # W3-P10: di-/polynuclear oxoacid anion word ('diphosphate')
        # BEFORE the generic name_anion, which would otherwise emit the neutral acid
        # name ('diphosphoric acid') and collapse the salt to the unsupported fallback.
        from .inorganic_acids import name_polyacid_anion
        poly_anion = name_polyacid_anion(frag_mol)
        if poly_anion:
            anion_names.append(poly_anion)
            continue
        # Organic anion: chokepoint first (the sound parent decision), then the
        # legacy name_anion path on '' (retained carboxylate names etc.).
        name = route_charged(frag_mol, style) or name_anion(frag_mol, style)
        _be = (general_fallback or general_fallback_unverified
               or allow_aromatic_general)
        # abstain-recovery (mirrors the cation branch): a confidently-wrong
        # ordinary anion name would shadow the best-effort per-component namer.
        # Under a best-effort flag, verify the ordinary anion name round-trips to
        # THIS fragment; replace with the (RT-verified) best-effort name only
        # when that yields a verified alternative, so a correct ordinary name is
        # never discarded (PIN tier + canary salts stay byte-identical).
        if _be and name and not _ion_fragment_roundtrips(frag_mol, name):
            alt = _name_organic_ion_best_effort(
                smiles, general_fallback=general_fallback,
                general_fallback_unverified=general_fallback_unverified,
                allow_aromatic_general=allow_aromatic_general)
            if alt:
                name = alt
        if not name:
            # -1b: best-effort fallback for a complex organic anion the
            # ordinary namer cannot build. Best-effort-gated -> PIN tier unchanged.
            name = _name_organic_ion_best_effort(
                smiles, general_fallback=general_fallback,
                general_fallback_unverified=general_fallback_unverified,
                allow_aromatic_general=allow_aromatic_general)
        if name:
            anion_names.append(name)
        # Skip unnamed anions rather than using generic 'anion'

    # fail-closed (0-wrong): every anion fragment must have produced a
    # name. Silently dropping an unnameable anion (e.g. a chlorosilanolate) and
    # joining only the subset that DID name is a silent atom-drop -- abstain
    # (return '') instead, mirroring the cation-loop guard above (abort-whole
    # -- name every fragment or emit nothing).
    if len(anion_names) != len(frags['anions']):
        return ''

    # a phase Lever A2 (0-wrong preserved): a genuine NEUTRAL fragment
    # reaching this point is unaccounted for. The only two places a neutral
    # fragment is legitimately consumed are the H+-merge hydroacid-salt
    # branch above (:294-318, which RETURNS directly on success) and -- not
    # applicable here -- a zwitterion, which `parse_salt_fragments`
    # (ions.py:551) buckets by WHOLE-FRAGMENT net formal charge, so a
    # net-neutral zwitterion is a SINGLE fragment with no separate
    # cation/anion entries and never reaches `name_salt` at all (`is_salt`
    # requires both `cations` and `anions` non-empty). So any survivor in
    # `neutrals` here is a real extraneous organic/inorganic co-fragment.
    #
    # Previously this ALWAYS aborted (mirroring the cation/anion guards
    # above, an abort-whole guard) -- but that
    # made a recognized water of crystallization (e.g. cetylpyridinium
    # chloride monohydrate, CHEBI:3566) abstain even though the ionic part
    # names cleanly. general nomenclature explicitly allows a
    # water solvate to be folded as a "<name> <mult>hydrate" suffix (BB
    # line 4657/4685: "...monohydrate"), so fold a WATER-ONLY neutral set
    # into that suffix and keep the fail-closed abstain for any OTHER
    # (unrecognized) neutral co-former -- never silently drop it or force
    # a wrong/partial name.
    from .adducts import _HYDRATE_MULTIPLIERS
    solvate_suffix = ''
    if neutrals:
        water = [n for n in neutrals
                 if Chem.MolToSmiles(n['mol'], canonical=True) == 'O']
        other = [n for n in neutrals
                 if Chem.MolToSmiles(n['mol'], canonical=True) != 'O']
        if other:
            return ''  # unrecognized neutral co-former -> fail closed (0-wrong)
        w = len(water)
        hydrate_prefix = _HYDRATE_MULTIPLIERS.get(w)
        if hydrate_prefix is None:
            return ''  # water count outside the mono..deca table -> fail closed
        solvate_suffix = ' ' + hydrate_prefix + 'hydrate'

    # --- Hydrogen prefix for partial salts (IUPAC ---
    # When an anion fragment still has protonated carboxylic acid groups
    # (-COOH), it is only partially deprotonated. Insert "hydrogen"
    # between cation and anion names.
    # E.g., "sodium hydrogen fumarate" = one Na+ + one COOH + one COO-.
    hydrogen_prefix = ''
    if len(frags['anions']) == 1 and cation_names:
        anion_frag = frags['anions'][0]
        protonated_acids = _count_protonated_acid_sites(anion_frag['mol'])
        # Do NOT insert the method-(2) 'hydrogen' word when the anion name already
        # expresses the acidic hydrogen(s): method (1) organic salts carry a
        # 'carboxy' prefix ('potassium 6-carboxyhexanoate',, and the
        # inorganic acid-anion words already embed 'hydrogen' ('hydrogen carbonate',
        # 'dihydrogen phosphate',. Only a fully-systematic organic
        # '-oate/-dioate' anion takes the separate 'hydrogen' (method 2 general
        # form, e.g. 'sodium hydrogen but-2-enedioate').
        acid_h_expressed = any(('carboxy' in a) or ('hydrogen' in a)
                               for a in anion_names)
        if protonated_acids > 0 and not acid_h_expressed:
            hydrogen_prefix = _HYDROGEN_PREFIXES.get(
                protonated_acids, 'hydrogen'
            )

    # Handle stoichiometry - count duplicates
    cation_counts = Counter(cation_names)
    anion_counts = Counter(anion_names)

    # Format cation part with multipliers
    formatted_cations = []
    for name in sorted(cation_counts.keys()):
        count = cation_counts[name]
        formatted_cations.append(_apply_stoichiometric_prefix(name, count))

    # Format anion part with multipliers
    formatted_anions = []
    for name in sorted(anion_counts.keys()):
        count = anion_counts[name]
        formatted_anions.append(_apply_stoichiometric_prefix(name, count))

    # Combine: cations first, then hydrogen prefix (if any), then anions
    if hydrogen_prefix and formatted_anions:
        result_parts = formatted_cations + [hydrogen_prefix] + formatted_anions
    else:
        result_parts = formatted_cations + formatted_anions

    result = ' '.join(result_parts)
    return result + solvate_suffix


def _apply_stoichiometric_prefix(name: str, count: int) -> str:
    """
    Apply di-, tri-, tetra- prefix for stoichiometry.

    Args:
        name: Base ion name
        count: Number of occurrences

    Returns:
        Name with stoichiometric prefix if count > 1

    Example:
        >>> _apply_stoichiometric_prefix('acetate', 2)
        'diacetate'
        >>> _apply_stoichiometric_prefix('sodium', 1)
        'sodium'
        >>> _apply_stoichiometric_prefix('D-gluconate', 2)
        'bis(D-gluconate)'
    """
    if count == 1:
        return name

    if _ion_needs_enclosing_multiplier(name):
        word = COMPLEX_STOICHIOMETRIC_PREFIXES.get(count, f"{count}kis")
        return f"{word}({name})"

    prefix = STOICHIOMETRIC_PREFIXES.get(count, str(count))
    return f"{prefix}{name}"


# === ZWITTERION NAMING ===

def name_zwitterion(mol, style: str = 'pin') -> str:
    """
    Name a zwitterionic compound.

    IUPAC rules:
    - Anionic centers get lower locants (higher seniority)
    - Cationic suffixes cited BEFORE anionic suffixes
    - Format: base-name-cation_suffix-anion_suffix

    Common zwitterions:
    - Amino acid zwitterions: glycine = 2-ammonioacetate (PIN) or glycine (trivial)

    Args:
        mol: RDKit Mol object with internal positive and negative charges
        style: 'pin' for preferred names

    Returns:
        Zwitterion IUPAC name

    Example:
        >>> mol = Chem.MolFromSmiles('C[N+](C)(C)CC(=O)[O-]')
        >>> name_zwitterion(mol)
        '(trimethylazaniumyl)acetate'
    """
    if mol is None:
        return ''

    # Check for retained amino acid names first (: amino-acid zwitterions +
    # betaines sequenced first). These (glycine / L-alanine /...) are valid
    # OPSIN-parseable retained names per (which allows retained names).
    if style != 'systematic':
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in RETAINED_AMINO_ACID_ZWITTERIONS:
            return RETAINED_AMINO_ACID_ZWITTERIONS[canonical]

    # GUARD 4: the route_charged chokepoint owns the anion-is-parent
    # override + the structured (…azaniumyl) cation prefix. This
    # REPLACES the deleted hardcoded "2-azaniumyl{base}" / "betaine" / "ammonium
    # {base}" f-string band-aids (fix-methodology.md: structured, not literal).
    from .charged_router import route_charged
    routed = route_charged(mol, style)
    if routed:
        return routed

    # charged Slice B, correcting the 4782742f over-reach): the
    # NEUTRAL-form name of a zwitterion is NOT its PIN. (the Blue Book
    #:1779 item (e)) makes the IONIC form the PIN — the anion is the parent and
    # each protonated-amine cation is an ``azaniumyl`` prefix — and that form is
    # built by ``route_charged`` GUARD 4 (called just above:
    # ``_name_primary_amine_azaniumyl_zwitterion``). The neutral-form namers below
    # are therefore a BEST-EFFORT-ONLY fallback (a non-PIN but full-InChIKey
    # RT-correct systematic name, reachable only when the azaniumyl PIN could not
    # be constructed). On the DEFAULT / PIN tier the contract is azaniumyl-or-
    # ABSTAIN: returning the neutral name there would ship a non-PIN name on the
    # PIN path (the 4782742f defect: S-methylcysteine zwitterion ->
    # ``(2R)-2-amino-3-(methylsulfanyl)propanoic acid``). Tier read from
    # ``best_effort_ctx`` (the same signal ``charged_router._reenter`` reads;
    # published by the namer, default False -> PIN tier).
    from ..metrics.provenance import best_effort_ctx
    if not best_effort_ctx.get():
        return ''  # DEFAULT / PIN tier: azaniumyl PIN (above) or abstain

    # --- BEST-EFFORT tier only: non-PIN neutral-form fallback ----------------
    # A net-zero amino-acid-shaped zwitterion neutralizes IN PLACE to the SAME
    # full InChIKey (InChI's mobile-H tautomer perception), so the neutral name
    # still round-trips; it still faces the caller's E1/ gate. This
    # degrades a table/PIN miss to an uglier systematic name rather than to
    # silence (T4: an abstention is a defect), but only above the PIN tier.
    if _is_amino_acid_zwitterion(mol):
        aa_name = _name_amino_acid_zwitterion(mol, style)
        if aa_name:
            return aa_name
    return _name_general_zwitterion(mol, style)


def _is_amino_acid_zwitterion(mol) -> bool:
    """
    Check if molecule is amino acid zwitterion [NH3+]-Cn-[COO-].

    Detects alpha, beta, and gamma amino acid zwitterion patterns
    per IUPAC.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule matches alpha/beta/gamma amino acid zwitterion pattern

    Example:
        >>> mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        >>> _is_amino_acid_zwitterion(mol)
        True
        >>> mol = Chem.MolFromSmiles('[NH3+]CCC([O-])=O')
        >>> _is_amino_acid_zwitterion(mol)
        True
    """
    # Check alpha, beta, and gamma patterns
    for smarts in (ALPHA_AA_ZWITTERION, BETA_AA_ZWITTERION, GAMMA_AA_ZWITTERION):
        pattern = _compiled_smarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return True

    return False


def _name_amino_acid_zwitterion(mol, style: str) -> str:
    """
    Name amino acid zwitterion (e.g., glycine zwitterion).

    Per IUPAC recommendation, amino acid zwitterions may be named as their
    neutral form (e.g., "2-aminoacetic acid" for glycine zwitterion). OPSIN
    parses these neutral-form names correctly, and they are RT-correct at
    connectivity (InChI-L1 ignores charge).

    The structured ionic form ((azaniumyl)…oate) is produced UPSTREAM
    by route_charged GUARD 4 (called first in name_zwitterion); this function is
    only reached when the chokepoint declined, so it returns the neutral form or
    '' (honest-fail) — the carbon-counting "2-azaniumyl{base}" f-string band-aid
    was DELETED (169.6-04, fix-methodology.md).

    Args:
        mol: RDKit Mol object
        style: 'pin' for systematic, others may use trivial

    Returns:
        Neutral amino acid name, or '' on failure (NO carbon-counting fallback).
    """
    # Neutralize and name the neutral form (the neutral-form recommendation).
    neutral_mol = _neutralize_zwitterion(mol)
    if neutral_mol is not None:
        try:
            neutral_smiles = Chem.MolToSmiles(neutral_mol, canonical=True)
            if neutral_smiles:
                # W8 P3 fail-closed veto: the
                # "convenient neutral form" dispensation is licensed ONLY for the
                # monoamino monocarboxylic acids RETAINED IN TABLE 10.4 (the 20
                # canonical STANDARD_AMINO_ACIDS). BB's own example
                # (S-methyl-L-cysteine zwitterion) shows a non-standard /
                # substituted amino acid's zwitterion PIN is the Method-1 ionic
                # form '(2S)-2-azaniumyl-3-(methylsulfanyl)propanoate', NOT the
                # neutral '...oic acid' -- naming it as the plain neutral acid
                # here would silently DROP the ionization state and describe a
                # different (neutral) species than the input. The charged
                # Method-1 engine is a later phase; fail closed rather than emit
                # that wrong-structure leak. Standard AAs (glycine/alanine/...)
                # are unaffected -- their neutral skeleton IS the bare Table 10.4
                # retained name, which explicitly sanctions.
                from ..data.amino_acids import is_standard_amino_acid
                nostereo_smi = Chem.MolToSmiles(
                    neutral_mol, isomericSmiles=False, canonical=True
                )
                nostereo_mol = Chem.MolFromSmiles(nostereo_smi)
                is_bare_standard_aa = nostereo_mol is not None and (
                    is_standard_amino_acid(
                        Chem.MolToSmiles(nostereo_mol, canonical=True)
                    )
                )
                if not is_bare_standard_aa:
                    return ''  # fail closed -- not a Table-10.4 retained AA

                from ..assembly.fragment_naming import name_fragment_recursively
                neutral_name = name_fragment_recursively(neutral_smiles)
                if neutral_name and neutral_name != 'zwitterion':
                    return neutral_name
        except (RecursionError, ValueError, RuntimeError):
            pass

    # Honest-fail (no carbon-counting band-aid): the structured route_charged
    # GUARD-4 path ran first; if both declined there is no valid name here.
    return ''


# === ZWITTERION NEUTRALIZATION HELPERS ===


def _neutralize_zwitterion(mol):
    """
    Neutralize a zwitterion by removing internal charges.

    Handles:
    - Protonated amines ([NH3+] -> NH2): reduce explicit H by charge
    - Quaternary ammonium ([N+](C)(C)(C)C): skip (can't neutralize without
      breaking a bond - not chemically meaningful as neutral)
    - Deprotonated acids ([COO-] -> COOH): increase explicit H by abs(charge)

    Args:
        mol: RDKit Mol object with internal charges

    Returns:
        Neutralized RDKit Mol object, or None on failure
    """
    try:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge > 0:
                cur_h = atom.GetNumExplicitHs()
                total_h = atom.GetTotalNumHs()
                if total_h >= charge:
                    # Protonated: remove H to compensate
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(max(0, cur_h - charge))
                else:
                    # Quaternary (no H to remove): just drop charge
                    # This may create an invalid valence; will be caught by sanitize
                    atom.SetFormalCharge(0)
                    atom.SetNoImplicit(True)
            elif charge < 0:
                atom.SetFormalCharge(0)
                cur_h = atom.GetNumExplicitHs()
                atom.SetNumExplicitHs(cur_h + abs(charge))

        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def _name_as_neutral(mol, style: str) -> str:
    """
    Try to name a zwitterion by neutralizing it first.

    Strips all internal charges, names the neutral form using the
    standard naming pipeline. This is an acceptable approximation
    per IUPAC for complex zwitterions.

    Args:
        mol: RDKit Mol with zwitterionic charges
        style: Naming style

    Returns:
        Name of the neutral form, or empty string on failure
    """
    neutral = _neutralize_zwitterion(mol)
    if neutral is None:
        return ''

    try:
        neutral_smiles = Chem.MolToSmiles(neutral, canonical=True)
        if not neutral_smiles:
            return ''

        from ..assembly.fragment_naming import name_fragment_recursively
        neutral_name = name_fragment_recursively(neutral_smiles)
        # Guard: never return 'zwitterion' from the neutral naming path
        if neutral_name and neutral_name != 'zwitterion':
            return neutral_name
    except (RecursionError, ValueError, RuntimeError):
        pass

    return ''


# === GENERAL ZWITTERION NAMING ===


def _name_general_zwitterion(mol, style: str) -> str:
    """
    Name a general zwitterion (not amino acid pattern).

    For zwitterions with various functional groups, combines
    the cationic and anionic descriptors. Falls back to naming
    the neutralized form if specific pattern matching fails.

    Args:
        mol: RDKit Mol object
        style: Naming style

    Returns:
        Zwitterion name, or empty string if naming fails.
        Never returns the literal 'zwitterion'.
    """
    sites = get_ion_sites(mol)

    cation_sites = sites.get('cations', [])
    anion_sites = sites.get('anions', [])

    if not cation_sites or not anion_sites:
        # No ionic sites found - cannot name as zwitterion
        return ''

    # Determine the type of cation and anion
    cation_element = cation_sites[0]['element'] if cation_sites else ''
    anion_element = anion_sites[0]['element'] if anion_sites else ''

    # Build name based on ionic sites
    if cation_element == 'N' and anion_element == 'O':
        # Likely amino acid-like or betaine-like
        result = _infer_zwitterion_name(mol, cation_sites, anion_sites)
        if result:
            return result

    # Fallback: neutralize and name the skeleton
    neutral_name = _name_as_neutral(mol, style)
    if neutral_name:
        return neutral_name

    # Honest failure instead of placeholder literal
    return ''


def _infer_zwitterion_name(
    mol,
    cation_sites: List[Dict[str, Any]],
    anion_sites: List[Dict[str, Any]]
) -> str:
    """
    Infer zwitterion name from ion site positions.

    Analyzes the molecular structure to determine appropriate naming.

    Args:
        mol: RDKit Mol object
        cation_sites: List of cation site dictionaries
        anion_sites: List of anion site dictionaries

    Returns:
        Inferred zwitterion name, or empty string on failure.

    169.6-04: the hardcoded ``betaine`` SMARTS literal and the carbon-counting
    ``ammonium {base}`` f-string band-aids were DELETED. The structured
     form is produced UPSTREAM by route_charged GUARD 4 (called first in
    name_zwitterion); this function is only reached when the chokepoint declined,
    so it returns the neutral-form name or '' (honest-fail, fix-methodology.md).
    """
    # Neutralize and name the parent amino compound (the neutral-form
    # recommendation for amino-acid-shaped zwitterions). No literal, no
    # carbon-counting fallback.
    neutral_name = _name_as_neutral(mol, 'pin')
    if neutral_name:
        return neutral_name

    return ''


# === SALT DETECTION HELPERS ===

def is_salt(mol) -> bool:
    """
    Check if a molecule is a salt (has separate cation and anion fragments).

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a salt

    Example:
        >>> mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        >>> is_salt(mol)
        True
    """
    if mol is None:
        return False

    # a phase-a: a genuine salt is charge-balanced overall; see the
    # matching guard (and its rationale) in name_salt above.
    if Chem.GetFormalCharge(mol) != 0:
        return False

    frags = parse_salt_fragments(mol)
    return bool(frags['cations'] and frags['anions'])


def is_zwitterion(mol) -> bool:
    """
    Check if a molecule is a zwitterion (internal + and - charges).

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a zwitterion

    Example:
        >>> mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        >>> is_zwitterion(mol)
        True
    """
    if mol is None:
        return False

    # Zwitterion: single fragment with both + and - charges that cancel
    frags = Chem.GetMolFrags(mol, asMols=True)

    if len(frags) != 1:
        return False

    # Check for both positive and negative atoms
    has_positive = False
    has_negative = False

    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            has_positive = True
        elif charge < 0:
            has_negative = True

    if not (has_positive and has_negative):
        return False

    # Net charge should be zero
    net_charge = Chem.GetFormalCharge(mol)
    return net_charge == 0
