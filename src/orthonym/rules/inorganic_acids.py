"""Free inorganic oxoacids (P-67 / P-65.2.1) — retained / preselected PINs.

Orthonym has no functional-parent subsystem for the mononuclear and simple
polynuclear inorganic oxoacids: free ``phosphoric``/``sulfuric``/… either return
``unknown`` or mis-name (``OP(=O)(O)O`` → ``trihydrophosphate``; ``OC(=O)O`` →
``methane``). The Blue Book gives these *retained / preselected* names directly
(P-67.1.1 phosphoric acid, sulfuric acid; P-65.2.1 carbonic acid; P-68.2 silicic
acid; P-67.2.1 di-acids), so there is no constitutional algorithm to run — the
correct PIN is a table lookup.

This module is an EXACT canonical-SMILES recognizer: it returns the retained PIN
only for the precise structures in ``_INORGANIC_OXOACIDS`` and ``None`` for
everything else (fail-closed). Because the match is on the full-molecule RDKit
canonical SMILES, there are zero false positives — a charged conjugate base, an
ester, or any substituted derivative simply will not match and cascades onward.

DEFERRED (A10 honest-fail-on-data, NOT in this build): organyl-substituted
oxoacids (``CCP(=O)(O)O`` → ethylphosphonic acid), acyl-halide word-forms
(phosphoryl/sulfuryl), and end-to-end polyacid / di-/triphosphate-*ester*
numbering. Those need a functional-replacement engine; emitting them here would
risk wrong names, so the conjugate-controller P-O-P out-of-scope gate is left in
place and only the FREE acids are tabled (free di-acids included as exact rows).
"""
from typing import Optional

from rdkit import Chem

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
}


def name_inorganic_acid(mol) -> Optional[str]:
    """Return the retained PIN for a free inorganic oxoacid, else ``None``.

    Pure: no mol mutation, no global state. Recomputes the RDKit canonical
    SMILES so the key matches regardless of how the input was written.
    """
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception:
        return None
    return _INORGANIC_OXOACIDS.get(canonical)
