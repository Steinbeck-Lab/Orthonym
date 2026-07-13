"""Inorganic oxoacids + functional-class derivatives (P-67 / P-65.2.1) — PINs.

Orthonym has no functional-parent subsystem for the mononuclear and simple
polynuclear inorganic oxoacids: free ``phosphoric``/``sulfuric``/… either return
``unknown`` or mis-name (``OP(=O)(O)O`` → ``trihydrophosphate``; ``OC(=O)O`` →
``methane``). The Blue Book gives these *retained / preselected* names directly
(P-67.1.1 phosphoric acid, sulfuric acid; P-65.2.1 carbonic acid; P-68.2 silicic
acid; P-68.1.4.1 boric/boronic/borinic acid; P-67.2.1 di-acids), so there is no
constitutional algorithm to run — the correct PIN is a table lookup.

This module is an EXACT canonical-SMILES recognizer: it returns the PIN only for
the precise structures tabled here and ``None`` for everything else (fail-closed).
Because the match is on the full-molecule RDKit canonical SMILES, there are zero
false positives — a charged conjugate base, an ester, or any substituted
derivative simply will not match and cascades onward.

Three tables, all consulted by :func:`name_inorganic_acid`:
  * ``_INORGANIC_OXOACIDS`` — the FREE acids (P-67.1.1 / P-65.2.1 / P-68).
  * ``_INORGANIC_ACID_DERIVATIVES`` — v23 Phase 9: acid-halide acyl-word forms
    (``phosphoryl trichloride``, ``sulfuryl dichloride``, P-67.1.2.5.1) and the
    amide functional-class names (``phosphoric triamide``, ``sulfuric diamide``,
    ``sulfamic acid``, P-67.1.2.6.1). The acyl-halide names are spelled by the
    shared FRN engine (``rules.functional_replacement``).
  * ``_CARBONIC_FRN`` — v23 Phase 9: the carbonic/carbamic functional-replacement
    acids (P-65.2.1.2/.3) — ``carbonoperoxoic`` / ``carbonodithioic`` /
    ``carbonotrithioic`` / ``carbonimidic`` / ``carbamimidic`` / ``dicarbonic`` /
    ``tricarbonic`` — names built by the SAME FRN engine.

DEFERRED (Phase 19, name-exact gold — OPSIN rejects the word-form so they cannot
be round-trip-validated): the italic O/S/Se tautomer-locant acid words
(``carbonothioic S-acid``, ``carbamothioic O-acid``, ``phosphorothioic O,O-acid``).
Also out of scope: end-to-end di-/triphosphate-*ester* numbering.
"""
from typing import Optional

from rdkit import Chem

from .functional_replacement import (
    build_acyl_halide_name,
    build_frn_acid_name,
    build_polyacid_name,
)

# Keyed by RDKit ``Chem.MolToSmiles`` canonical SMILES of the neutral, fully
# protonated acid. Every name has been confirmed to round-trip through OPSIN to
# the same structure. Verify any new row against PIN-VERIFICATION before adding.
_INORGANIC_OXOACIDS = {
    "O=P(O)(O)O": "phosphoric acid",            # P-67.1.1.1  H3PO4 (preselected)
    "O=S(=O)(O)O": "sulfuric acid",             # P-67.1.1.1  H2SO4
    "O=S(O)O": "sulfurous acid",                # P-67.1.1    H2SO3
    "O=C(O)O": "carbonic acid",                 # P-65.2.1    H2CO3 (retained)
    "O[Si](O)(O)O": "silicic acid",             # P-68.2      Si(OH)4
    # (nitric acid HNO3 is intentionally NOT here — it already round-trips via the
    #  RETAINED_NAME tier; zwitterion/salt predicates decline its charge-separated
    #  form, so no early mis-route. Kept out to avoid a duplicate-table drift.)
    "O=P(O)(O)OP(=O)(O)O": "diphosphoric acid",  # P-67.2.1   (HO)2P(O)-O-P(O)(OH)2
    "O=S(=O)(O)OS(=O)(=O)O": "disulfuric acid",  # P-67.2.1   (HO)SO2-O-SO2(OH)
    # --- v23 Phase 7: mononuclear halogen oxoacids (P-67.1.1.1, all preselected
    #     PINs). RDKit canonicalises the hypervalent X(=O)n(OH) forms to a
    #     charge-separated SMILES, so the keys carry the [X+n]/[O-] charges; the
    #     molecule is neutral overall and reaches INORGANIC_ACID@40 (the dispatch
    #     minimum). Every name OPSIN-RT-confirmed. ---
    "OCl": "hypochlorous acid",                  # P-67.1.1.1  Cl(OH)
    "[O-][Cl+]O": "chlorous acid",               # P-67.1.1.1  Cl(O)(OH)
    "[O-][Cl+2]([O-])O": "chloric acid",         # P-67.1.1.1  Cl(O)2(OH)
    "[O-][Cl+3]([O-])([O-])O": "perchloric acid",  # P-67.1.1.1 Cl(O)3(OH)
    "OBr": "hypobromous acid",                   # P-67.1.1.1  Br(OH)
    "[O-][Br+]O": "bromous acid",                # P-67.1.1.1  Br(O)(OH)
    "[O-][Br+2]([O-])O": "bromic acid",          # P-67.1.1.1  Br(O)2(OH)
    "[O-][Br+3]([O-])([O-])O": "perbromic acid",  # P-67.1.1.1 Br(O)3(OH)
    "OI": "hypoiodous acid",                      # P-67.1.1.1  I(OH)
    "[O-][I+]O": "iodous acid",                   # P-67.1.1.1  I(O)(OH)
    "[O-][I+2]([O-])O": "iodic acid",            # P-67.1.1.1  I(O)2(OH)
    "[O-][I+3]([O-])([O-])O": "periodic acid",   # P-67.1.1.1  I(O)3(OH) (metaperiodic)
    # --- v23 Phase 8: the three preselected/retained boron parent acids
    #     (P-68.1.4.1 / P-67.1.1.1). These are the UNSUBSTITUTED parents; the
    #     carbon-bearing R-boronic acids (CB(O)O -> methylboronic acid) carry a
    #     carbon and so never match these exact carbon-free keys — they continue
    #     to the dedicated boronic_acid handler. Boron is kept OUT of the Group-14
    #     substitutive-suffix path (organometallics._GROUP14_SUFFIX_ELEMENTS) — a
    #     boron hydroxy acid is named here, never 'boranetriol' (the v22 G2 lesson).
    #     Every name OPSIN-RT-confirmed. ---
    "OB(O)O": "boric acid",                      # P-68.1.4.1  B(OH)3   (H3BO3)
    "OBO": "boronic acid",                       # P-68.1.4.1  HB(OH)2  (H3BO2, parent)
    "BO": "borinic acid",                        # P-68.1.4.1  H2B(OH)  (H3BO,  parent)
}

# --- v23 Phase 9: acid-halide + amide functional-class derivatives ---
# Acid halides of phosphoric/sulfuric (identical replaceable -OH groups) use the
# acyl-group word (P-67.1.2.5.1); amides replace all -OH by -NH2 (P-67.1.2.6.1).
# These intercept @40 BEFORE the OPSIN-imported retained tier (RETAINED_NAME@1300)
# which would otherwise emit the non-PIN 'phosphorous(v) oxychloride' /
# 'phosphoramide' / 'sulfamide'. Keys are RDKit canonical SMILES of the neutral
# molecule; every name is OPSIN-RT-confirmed.
_INORGANIC_ACID_DERIVATIVES = {
    "O=P(Cl)(Cl)Cl": build_acyl_halide_name("phosphoryl", "chloride", 3),       # POCl3
    "O=P(F)(F)F": build_acyl_halide_name("phosphoryl", "fluoride", 3),          # POF3
    "O=P(Br)(Br)Br": build_acyl_halide_name("phosphoryl", "bromide", 3),        # POBr3
    "S=P(Cl)(Cl)Cl": build_acyl_halide_name("phosphorothioyl", "chloride", 3),  # PSCl3 (FRN thio)
    "O=S(=O)(Cl)Cl": build_acyl_halide_name("sulfuryl", "chloride", 2),         # SO2Cl2
    "O=S(=O)(F)F": build_acyl_halide_name("sulfuryl", "fluoride", 2),           # SO2F2
    "NP(N)(N)=O": "phosphoric triamide",   # P-67.1.2.6.1 PIN ('phosphoramide' = non-PIN alt)
    "NS(N)(=O)=O": "sulfuric diamide",     # P-67.1.2.6.1 PIN ('sulfamide' = general name)
    # Wave-2 completion C2 (carbonic-family composite-N parents; exact keys):
    "N=C(N)SSC(=N)N": "carbamimidic dithioperoxyanhydride",  # P-66.4.1.5 (formamidine disulfide; anhydride is the senior class per P-41)
    "NN=C(N)N": "carbonohydrazonic diamide",                 # P-66.4.2.2 (aminoguanidine, BB verbatim)
    "NN=C(NN)NN": "hydrazinecarbohydrazonohydrazide",        # P-66.4.3.3 (BB verbatim)
    "NS(=O)(=O)O": "sulfamic acid",        # P-67.1.2.4.1.1 H2N-SO2-OH (contraction of sulfuramidic)
    # Wave-2 P1AM (2026-07-09): hydrazine-parent carbonic-family parents.
    # P-66.1.1.1.1.3 (BB 32675): 'carboxamide' is ALWAYS the suffix on a
    # heteroacyclic parent -> hydrazinecarboxamide (PIN); P-68.3.1.2.4
    # (BB 38623): "The systematic name is the preferred IUPAC name"
    # (semicarbazide = general nomenclature only). Intercepts @40 before
    # RETAINED_NAME@1300 which emitted 'semicarbazide'.
    "NNC(N)=O": "hydrazinecarboxamide",
    # P-66.4.2.2 (BB 34480 verbatim): amidrazone of carbonic acid.
    "N=C(NN)NN": "hydrazinecarboximidohydrazide",
    # P-66.4.2.2 family (aminoguanidine =NH tautomer; hydrazine parent +
    # carboximidamide suffix per P-66.4.1.1; OPSIN-RT verified).
    "N=C(N)NN": "hydrazinecarboximidamide",
    # P-66.4.1.2.1.2 (BB 34298): "The names biguanide, triguanide, etc., are
    # no longer recommended. Condensed guanidines ... are named systematically
    # as the diamides of imidodicarbonimidic acid". Bare parent only; the
    # @40 exact key never matches substituted forms, which keep the guanidine
    # handler's RT-valid general name until the N^n locant subsystem exists.
    "N=C(N)NC(=N)N": "imidodicarbonimidic diamide",
}

# --- v23 Phase 9: carbonic/carbamic functional-replacement acids ---
# Plain (non-italic-locant) forms that OPSIN round-trips; names built by the
# shared FRN engine. The single-chalcogen tautomer forms needing italic S-/O-
# acid locants (carbonothioic S-acid) are DEFERRED to Phase 19 (no OPSIN RT).
_CARBONIC_FRN = {
    "O=C(O)OO": build_frn_acid_name("carbon", "peroxo", 1),          # carbonoperoxoic acid
    "O=C(S)S": build_frn_acid_name("carbon", "thio", 2),             # carbonodithioic acid  (HS-CO-SH)
    "S=C(S)S": build_frn_acid_name("carbon", "thio", 3),             # carbonotrithioic acid (HS-CS-SH)
    "N=C(O)O": build_frn_acid_name("carbon", "imido", 1),            # carbonimidic acid     (HO-C(=NH)-OH)
    "N=C(N)O": build_frn_acid_name("carbam", "imido", 1),            # carbamimidic acid     (H2N-C(=NH)-OH)
    "O=C(O)OC(=O)O": build_polyacid_name("carbonic acid", 2),        # dicarbonic acid
    "O=C(O)OC(=O)OC(=O)O": build_polyacid_name("carbonic acid", 3),  # tricarbonic acid
}

# --- W3-P05 (P-65.1.8.1): carbonic-acid halides ---
# Carbonic acid (HO-CO-OH) with one -OH replaced by a halogen -> the retained
# ``carbono{halogen}idic acid`` PIN (P-65.1.8.1; BB 30684-30686: 'Cl-COOH
# carbonochloridic acid (PIN), not chloroformic acid'). P-65.1.8.1 (BB 30670)
# FORBIDS naming X-CO-OH as a substituted formic/methanoic acid, so the general
# chain-acid path's '1-chloro-1-oxomethanoic acid' is a non-PIN that MUST be
# intercepted @40. Exact full-molecule canonical-SMILES keys => zero false
# positives. Every name OPSIN-RT-confirmed.
_CARBONIC_ACID_HALIDES = {
    "O=C(O)Cl": "carbonochloridic acid",   # P-65.1.8.1  Cl-CO-OH
    "O=C(O)Br": "carbonobromidic acid",    # P-65.1.8.1  Br-CO-OH
    "O=C(O)F": "carbonofluoridic acid",    # P-65.1.8.1  F-CO-OH
    "O=C(O)I": "carbonoiodidic acid",      # P-65.1.8.1  I-CO-OH
}

# --- W3-P05 (P-65.1.8.2): C-substituted formic acids (retained-name base) ---
# Formic acid (H-CO-OH) whose formyl H is replaced by an approved substituent is
# named on the retained 'formic acid' parent. The ONLY such PIN is the nitro
# derivative (P-65.1.8.2; BB 30696: 'O2N-COOH nitroformic acid (PIN)') — this is
# NOT a general substituted-formic-acid engine (P-65.1.8.1 @30670 forbids that:
# the halides/pseudohalides are their own retained 'carbono...idic acid' PINs, not
# 'chloroformic'/'cyanoformic'). Exact full-molecule canonical-SMILES key =>
# zero false positives. Intercepts @40 before the general acid path; also
# preempts the (now-tightened) carbamic_acid SMARTS whose old form false-matched
# the nitro N and emitted a wrong 'carbamic acid'. OPSIN-RT-confirmed.
_C1_ACID_C_SUBSTITUTED = {
    "O=C(O)[N+](=O)[O-]": "nitroformic acid",   # P-65.1.8.2  O2N-CO-OH
}

# --- W3-P05 (P-65.2.1.4): carbonic-acid pseudohalides ---
# Carbonic acid (HO-CO-OH) with one -OH replaced by a pseudohalide (cyanido,
# azido) -> the retained ``carbono{pseudohalide}idic acid`` PIN (P-65.2.1.4
# @30814, cyanido infix; BB 30832 'NC-CO-OH carbonocyanidic acid (PIN)'; BB 30834
# carbonazidic acid). As with the halides, P-65.1.8.1 (@30670) FORBIDS the
# substituted-formic-acid name ('1-cyanomethanoic acid'). Exact full-molecule
# canonical-SMILES keys => zero false positives: cyanoacetic acid (N#CCC(=O)O)
# has an extra CH2 and never matches, cascading to the general acid path. Every
# name OPSIN-RT-confirmed.
_CARBONIC_ACID_PSEUDOHALIDES = {
    "N#CC(=O)O": "carbonocyanidic acid",        # P-65.2.1.4  NC-CO-OH
    "[N-]=[N+]=NC(=O)O": "carbonazidic acid",   # P-65.2.1.4  N3-CO-OH
}

# Merged lookup (no key overlap across the tables — distinct structures).
_ALL_INORGANIC = {
    **_INORGANIC_OXOACIDS,
    **_INORGANIC_ACID_DERIVATIVES,
    **_CARBONIC_FRN,
    **_CARBONIC_ACID_HALIDES,
    **_C1_ACID_C_SUBSTITUTED,
    **_CARBONIC_ACID_PSEUDOHALIDES,
}


def lookup_exact_acid_name(canonical_smiles: str) -> Optional[str]:
    """Return the retained / functional-replacement inorganic-acid PIN for an
    EXACT canonical-SMILES key (carbamimidic acid, carbonimidic acid, ...), or
    None. Consumed by the ester path so a tabled acid analog keeps its
    retained '-ic acid' stem instead of the systematic-but-non-PIN chain name
    (P-66.1.6.1.2.1: 'carbamimidic acid', not '1-aminomethanimidic acid')."""
    return _ALL_INORGANIC.get(canonical_smiles)


def name_silicate_ester(mol) -> Optional[str]:
    """Tetraalkyl silicate ester PIN (P-68.2.4 / BB 35978): a NEUTRAL silicon
    bearing exactly four ``-O-R`` groups (a fully-esterified silicic acid) ->
    ``{multiplier}{R} silicate`` (``tetramethyl silicate``,
    ``tetrakis(propan-2-yl) silicate``), or space-separated alphabetical citation
    for mixed R (``ethyl methyl ... silicate``). The functional-class ester word
    is the PIN, NOT the substitutive ``tetra(R)oxysilane``.

    Fail-closed (returns ``None``) for ANY Si that is not exactly Si(OR)4: a free
    Si-OH (partial ester / silicic acid -> stays the free-acid table row or the
    substitutive silanetriol path), an Si-C bond (genuine organosilicon ->
    P-69 / mononuclear-hydride), a charge, or a non-carbon-rooted O substituent.
    """
    if mol is None:
        return None
    si_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "Si"]
    if len(si_atoms) != 1:
        return None
    si = si_atoms[0]
    if si.GetFormalCharge() != 0 or si.GetDegree() != 4:
        return None
    si_idx = si.GetIdx()
    o_idxs = {nb.GetIdx() for nb in si.GetNeighbors()}
    if len(o_idxs) != 4:
        return None

    from ..assembly.substituent_enumerator import name_substituent

    r_names = []
    for o in si.GetNeighbors():
        # Each must be a neutral, H-free, divalent bridging O bonded to Si + one C.
        if (o.GetSymbol() != "O" or o.GetFormalCharge() != 0
                or o.GetTotalNumHs() != 0 or o.GetDegree() != 2):
            return None
        others = [nb for nb in o.GetNeighbors() if nb.GetIdx() != si_idx]
        if len(others) != 1 or others[0].GetSymbol() != "C":
            return None
        c = others[0]
        # BFS the R fragment (everything beyond the four Si-O bonds).
        frag, seen, stack = [], {si_idx, *o_idxs}, [c.GetIdx()]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            frag.append(x)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors()
                         if nb.GetIdx() not in seen)
        nm = name_substituent(mol, frag, c.GetIdx())
        if not nm:
            return None
        r_names.append(nm)
    if len(r_names) != 4:
        return None

    from collections import Counter
    from ..assembly.naming_utils import (
        get_multiplier_prefix, is_complex_substituent, alpha_sort_key,
    )
    counts = Counter(r_names)
    parts = []
    for nm in sorted(counts, key=alpha_sort_key):
        c = counts[nm]
        enclosed = f"({nm})" if is_complex_substituent(nm) else nm
        parts.append(enclosed if c == 1 else f"{get_multiplier_prefix(c, nm)}{enclosed}")
    return " ".join(parts) + " silicate"


def name_inorganic_acid(mol) -> Optional[str]:
    """Return the PIN for a free inorganic oxoacid or its tabled functional-class
    derivative (acid halide / amide / carbonic-FRN acid), the tetraalkyl silicate
    ester (P-68.2.4), else ``None``.

    Pure: no mol mutation, no global state. Recomputes the RDKit canonical
    SMILES so the key matches regardless of how the input was written.
    """
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception:
        return None
    tabled = _ALL_INORGANIC.get(canonical)
    if tabled is not None:
        return tabled
    # D-FOLLOWON item 10: tetraalkyl silicate ester (Si(OR)4). Routed here (@40)
    # so it intercepts BEFORE ORGANOMETALLIC@50 (which mis-claims Si as a metalloid
    # hub and linearizes the silyl-ester ligands into nonsense).
    return name_silicate_ester(mol)


def name_azinic_derivative(mol) -> Optional[str]:
    """P-61.5.3 ylidene derivatives of azinic acid H2N(O)OH (Wave-2
    completion C): CH3-CH=N(O)-OH -> ethylideneazinic acid (BB verbatim;
    aci-nitroethane). Fail-closed: exactly one N+ with a -O(-), an -OH and a
    DOUBLE bond to an unbranched all-carbon chain; net charge 0; nitro
    tautomers (=O on N) and nitronate anions never match. Pure."""
    from rdkit import Chem
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return None
    ns = [a for a in mol.GetAtoms()
          if a.GetSymbol() == 'N' and a.GetFormalCharge() == 1]
    if len(ns) != 1:
        return None
    n = ns[0]
    if n.IsInRing() or n.GetDegree() != 3:
        return None
    o_minus = oh = c = None
    for b in n.GetBonds():
        other = b.GetOtherAtom(n)
        if (other.GetSymbol() == 'O' and other.GetDegree() == 1
                and b.GetBondType() == Chem.BondType.SINGLE):
            if other.GetFormalCharge() == -1 and other.GetTotalNumHs() == 0:
                o_minus = other
            elif other.GetFormalCharge() == 0 and other.GetTotalNumHs() == 1:
                oh = other
        elif (other.GetSymbol() == 'C'
                and b.GetBondType() == Chem.BondType.DOUBLE):
            c = other
    if o_minus is None or oh is None or c is None:
        return None
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()
           if a.GetIdx() not in (n.GetIdx(), o_minus.GetIdx())):
        return None
    from ..assembly.naming_utils import unbranched_alkylidene_name
    ylidene = unbranched_alkylidene_name(mol, c.GetIdx(), n.GetIdx())
    if ylidene is None:
        return None
    if mol.GetNumHeavyAtoms() != 3 + sum(
            1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6):
        return None
    return f"{ylidene}azinic acid"
