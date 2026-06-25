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
    "NS(=O)(=O)O": "sulfamic acid",        # P-67.1.2.4.1.1 H2N-SO2-OH (contraction of sulfuramidic)
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

# Merged lookup (no key overlap across the three tables — distinct structures).
_ALL_INORGANIC = {**_INORGANIC_OXOACIDS, **_INORGANIC_ACID_DERIVATIVES, **_CARBONIC_FRN}


def name_inorganic_acid(mol) -> Optional[str]:
    """Return the PIN for a free inorganic oxoacid or its tabled functional-class
    derivative (acid halide / amide / carbonic-FRN acid), else ``None``.

    Pure: no mol mutation, no global state. Recomputes the RDKit canonical
    SMILES so the key matches regardless of how the input was written.
    """
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception:
        return None
    return _ALL_INORGANIC.get(canonical)
