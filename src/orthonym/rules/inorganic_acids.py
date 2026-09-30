"""Inorganic oxoacids + functional-class derivatives / — PINs.

Orthonym has no functional-parent subsystem for the mononuclear and simple
polynuclear inorganic oxoacids: free ``phosphoric``/``sulfuric``/… either return
``unknown`` or mis-name (``OP(=O)(O)O`` → ``trihydrophosphate``; ``OC(=O)O`` →
``methane``). The Blue Book gives these *retained / preselected* names directly
 phosphoric acid, sulfuric acid; carbonic acid; silicic
acid; boric/boronic/borinic acid; di-acids), so there is no
constitutional algorithm to run — the correct PIN is a table lookup.

This module is an EXACT canonical-SMILES recognizer: it returns the PIN only for
the precise structures tabled here and ``None`` for everything else (fail-closed).
Because the match is on the full-molecule RDKit canonical SMILES, there are zero
false positives — a charged conjugate base, an ester, or any substituted
derivative simply will not match and cascades onward.

Three tables, all consulted by:func:`name_inorganic_acid`:
  * ``_INORGANIC_OXOACIDS`` — the FREE acids / /.
  * ``_INORGANIC_ACID_DERIVATIVES`` — a phase: acid-halide acyl-word forms
    (``phosphoryl trichloride``, ``sulfuryl dichloride``, and the
    amide functional-class names (``phosphoric triamide``, ``sulfuric diamide``,
    ``sulfamic acid``,. The acyl-halide names are spelled by the
    shared FRN engine (``rules.functional_replacement``).
  * ``_CARBONIC_FRN`` — a phase: the carbonic/carbamic functional-replacement
    acids /.3) — ``carbonoperoxoic`` / ``carbonodithioic`` /
    ``carbonotrithioic`` / ``carbonimidic`` / ``carbamimidic`` / ``dicarbonic`` /
    ``tricarbonic`` — names built by the SAME FRN engine.

DEFERRED (a phase, name-exact gold — OPSIN rejects the word-form so they cannot
be round-trip-validated): the italic O/S/Se tautomer-locant acid words
(``carbonothioic S-acid``, ``carbamothioic O-acid``, ``phosphorothioic O,O-acid``).
Also out of scope: end-to-end di-/triphosphate-*ester* numbering.
"""
from typing import Optional

from rdkit import Chem

from ..perception.molcache import atoms_of, bonds_of  # audit 2026-09-03 (S2): per-call atom/bond tuples
from .functional_replacement import (
    build_acyl_halide_name,
    build_frn_acid_name,
    build_p_frn_acid_name,
    build_polyacid_name,
)

# Single-bonded halogen -> class-infix combining form (1)).
_HALIDO_INFIX = {"Cl": "chlorido", "F": "fluorido", "Br": "bromido", "I": "iodido"}

# a phase: central-atom stems for the class-infix FRN engine,
# WITHOUT the linking vowel (build_p_frn_acid_name appends it). Keyed by element,
# then by the skeletal (C + H) substituent count on the central atom: 0 -> the
# -oric parent stem, 1 -> the -onic parent stem (phosphinic/-inic FRN, count 2,
# is out of scope and fails closed). Extending the engine to As/Sb is exactly this
# table plus a central-atom search over {P, As, Sb}; build_p_frn_acid_name is
# unchanged.
#
# GUARD ("Name construction guidelines"): "The names phosphonous,
# phosphinous, phosphonic and phosphinic acid (and similarly for arsenic, antimony
#...) can only be used when P, As or Sb is attached to atoms of hydrogen, carbon
# or another atom of a parent hydride such as N, As, Si." Only C and H are counted
# as skeletal here, so a halogen / -N< amido / replaced -O on the central atom is a
# CLASS infix, never a skeletal substituent. Hence ClP(O)(OH)2 -> phosphorochloridic
# acid (skeletal 0 -> 'phosphor'), NOT chlorophosphonic acid; and Cl-As(O)(OH)2 ->
# arsorochloridic acid, not chloroarsonic acid. Any unrecognised neighbour fails the
# neighbour loop closed, so a P/As/Sb bond to a non-parent-hydride element is never
# silently absorbed into a wrong parent.
_FRN_CENTRAL_STEMS = {
    "P":  {0: "phosphor", 1: "phosphon"},
    "As": {0: "arsor", 1: "arson"},
    "Sb": {0: "stibor", 1: "stibon"},
}

# Keyed by RDKit ``Chem.MolToSmiles`` canonical SMILES of the neutral, fully
# protonated acid. Every name has been confirmed to round-trip through OPSIN to
# the same structure. Verify any new row against PIN-VERIFICATION before adding.
_INORGANIC_OXOACIDS = {
    "O=P(O)(O)O": "phosphoric acid",            # H3PO4 (preselected)
    "O=S(=O)(O)O": "sulfuric acid",             # H2SO4
    "O=S(O)O": "sulfurous acid",                # H2SO3
    "O=C(O)O": "carbonic acid",                 # H2CO3 (retained)
    "O[Si](O)(O)O": "silicic acid",             # Si(OH)4
    # (nitric acid HNO3 is intentionally NOT here — it already round-trips via the
    # RETAINED_NAME tier; zwitterion/salt predicates decline its charge-separated
    # form, so no early mis-route. Kept out to avoid a duplicate-table drift.)
    "O=P(O)(O)OP(=O)(O)O": "diphosphoric acid",  # (HO)2P(O)-O-P(O)(OH)2
    "O=P(O)(O)P(=O)(O)O": "hypodiphosphoric acid",  # (HO)2P(O)-P(O)(OH)2 (direct P-P; W3-P10 idx 1892 parent)
    "O=S(=O)(O)OS(=O)(=O)O": "disulfuric acid",  # (HO)SO2-O-SO2(OH)
    # --- a phase: mononuclear halogen oxoacids, all preselected
    # PINs). RDKit canonicalises the hypervalent X(=O)n(OH) forms to a
    # charge-separated SMILES, so the keys carry the [X+n]/[O-] charges; the
    # molecule is neutral overall and reaches INORGANIC_ACID@40 (the dispatch
    # minimum). Every name OPSIN-RT-confirmed. ---
    "OCl": "hypochlorous acid",                  # Cl(OH)
    "[O-][Cl+]O": "chlorous acid",               # Cl(O)(OH)
    "[O-][Cl+2]([O-])O": "chloric acid",         # Cl(O)2(OH)
    "[O-][Cl+3]([O-])([O-])O": "perchloric acid",  # Cl(O)3(OH)
    "OBr": "hypobromous acid",                   # Br(OH)
    "[O-][Br+]O": "bromous acid",                # Br(O)(OH)
    "[O-][Br+2]([O-])O": "bromic acid",          # Br(O)2(OH)
    "[O-][Br+3]([O-])([O-])O": "perbromic acid",  # Br(O)3(OH)
    "OI": "hypoiodous acid",                      # I(OH)
    "[O-][I+]O": "iodous acid",                   # I(O)(OH)
    "[O-][I+2]([O-])O": "iodic acid",            # I(O)2(OH)
    "[O-][I+3]([O-])([O-])O": "periodic acid",   # I(O)3(OH) (metaperiodic)
    # --- a phase: the three preselected/retained boron parent acids
    # /. These are the UNSUBSTITUTED parents; the
    # carbon-bearing R-boronic acids (CB(O)O -> methylboronic acid) carry a
    # carbon and so never match these exact carbon-free keys — they continue
    # to the dedicated boronic_acid handler. Boron is kept OUT of the Group-14
    # substitutive-suffix path (organometallics._GROUP14_SUFFIX_ELEMENTS) — a
    # boron hydroxy acid is named here, never 'boranetriol' (the lesson).
    # Every name OPSIN-RT-confirmed. ---
    "OB(O)O": "boric acid",                      # B(OH)3 (H3BO3)
    "OBO": "boronic acid",                       # HB(OH)2 (H3BO2, parent)
    "BO": "borinic acid",                        # H2B(OH) (H3BO, parent)
    # --- a phase A.0: complete the mononuclear preselected
    # free-acid table. "Names of mononuclear noncarbon oxoacids":
    # "Preselected names (see of the mononuclear noncarbon oxoacids
    # used for deriving preferred IUPAC names... are noted in the following
    # list". Every name below is verbatim "(preselected name)" in that list
    # — the CLOSED preselected table, so the correct PIN is a lookup, no
    # constitutional algorithm. Without these rows the engine either
    # abstained or emitted a right-molecule non-PIN skeletal name
    # (e.g. '2-hydroxy-1,3-dioxa-2-arsaprop-1-ene' for arsonic acid). Exact
    # full-molecule canonical-SMILES keys => zero false positives; every name
    # OPSIN-RT-confirmed (full InChIKey == input). The trivalent -ous /
    # substituted acids are producer work (A.1), not table rows.
    # Arsenic (As):
    "O=[AsH2]O": "arsinic acid",                 # H2As(O)(OH)
    "O[AsH2]": "arsinous acid",                  # H2As(OH)
    "O=[AsH](O)O": "arsonic acid",               # HAs(O)(OH)2
    "O[AsH]O": "arsonous acid",                  # HAs(OH)2
    "O=[As](O)(O)O": "arsoric acid",             # As(O)(OH)3 (preferred to 'arsenic acid')
    "O[As](O)O": "arsorous acid",                # As(OH)3 (formerly arsen(i)ous acid)
    # Nitrogen (N) oxoacids:
    "[O-][NH2+]O": "azinic acid",                # H2N(O)(OH)
    "[O-][NH+](O)O": "azonic acid",              # HN(O)(OH)2
    "ONO": "azonous acid",                       # HN(OH)2
    "ON(O)O": "azorous acid",                    # N(OH)3
    "[O-][N+](O)(O)O": "nitroric acid",          # N(O)(OH)3
    "O=NO": "nitrous acid",                      # HO-NO
    # Fluorine (F):
    "OF": "hypofluorous acid",                   # F(OH)
    # Phosphorus (P):
    "O=[PH2]O": "phosphinic acid",               # H2P(O)(OH)
    "OP": "phosphinous acid",                    # H2P(OH)
    "O=[PH](O)O": "phosphonic acid",             # HP(O)(OH)2
    "OPO": "phosphonous acid",                   # HP(OH)2
    "OP(O)O": "phosphorous acid",                # P(OH)3
    # Selenium (Se) / Tellurium (Te):
    "O=[Se](=O)(O)O": "selenic acid",            # Se(O)2(OH)2
    "O=[Se](O)O": "selenous acid",               # Se(O)(OH)2
    "O=[Te](=O)(O)O": "telluric acid",           # Te(O)2(OH)2
    # Antimony (Sb):
    "[O]=[SbH2][OH]": "stibinic acid",           # H2Sb(O)(OH)
    "[OH][SbH2]": "stibinous acid",              # H2Sb(OH)
    "[O]=[SbH]([OH])[OH]": "stibonic acid",      # HSb(O)(OH)2
    "[OH][SbH][OH]": "stibonous acid",           # HSb(OH)2
    "[O]=[Sb]([OH])([OH])[OH]": "stiboric acid",  # Sb(O)(OH)3 (preferred to 'antimonic acid')
    "[OH][Sb]([OH])[OH]": "stiborous acid",      # Sb(OH)3 (formerly antimonous acid)
    # --- a phase B.1: the di-/tri-nuclear PRESELECTED free acids.
    # "Preselected names": "The following traditional names are
    # retained as preselected names... for consistency... the numerical
    # infix 'di' has been uniformly used in naming dinuclear 'hypo' acids".
    # Every name below is verbatim "(preselected name)" in that list — a
    # CLOSED table, so the PIN is a lookup, no constitutional algorithm.
    # Currently each abstained ('inorganic/arsenic/antimony compound (not
    # supported)'). KEYS ARE THE Blue-Book PIN's OWN OPSIN-PARSE canonical
    # SMILES (the bb_conformance oracle key), so every name round-trips
    # through OPSIN to its exact key by construction => zero false positives.
    # Note: diarsonic / hypodiarsonic / distibonic / hypodistibonic (the
    # P/As-with-H '-onic' V-analogues) and hypodiboric and the polymeric
    # 'meta' acids are NOT here — OPSIN 2.9.0 parses them to a charged /
    # radical / unresolvable species, so they are not in the gold oracle and
    # cannot be RT-confirmed; they degrade to abstain (never a wrong name).
    # Boron / silicon:
    "OB(O)OB(O)O": "diboric acid",                      # (HO)2B-O-B(OH)2
    "O[Si](O)(O)O[Si](O)(O)O": "disilicic acid",        # (HO)3Si-O-Si(OH)3
    # Phosphorus di-nuclear. diphosphoric / hypodiphosphoric are above.
    "O=[PH](O)O[PH](=O)O": "diphosphonic acid",         # (HO)HP(O)-O-HP(O)(OH)
    "O=[PH](O)[PH](=O)O": "hypodiphosphonic acid",      # (HO)(O)HP-PH(O)(OH)
    "OPOPO": "diphosphonous acid",                      # HO-PH-O-PH-OH
    "OPPO": "hypodiphosphonous acid",                   # (HO)HP-PH(OH)
    "OP(O)OP(O)O": "diphosphorous acid",                # (HO)2P-O-P(OH)2
    "OP(O)P(O)O": "hypodiphosphorous acid",             # (HO)2P-P(OH)2
    # Arsenic di-nuclear. '-orous'/'-onous'/'-oric' families:
    "O[AsH]O[AsH]O": "diarsonous acid",                 # HO-AsH-O-AsH-OH
    "O[AsH][AsH]O": "hypodiarsonous acid",              # (HO)HAs-AsH(OH)
    "O=[As](O)(O)O[As](=O)(O)O": "diarsoric acid",      # (HO)2As(O)-O-As(O)(OH)2
    "O=[As](O)(O)[As](=O)(O)O": "hypodiarsoric acid",   # (HO)2(O)As-As(O)(OH)2
    "O[As](O)O[As](O)O": "diarsorous acid",             # (HO)2As-O-As(OH)2
    "O[As](O)[As](O)O": "hypodiarsorous acid",          # (HO)2As-As(OH)2
    # Antimony di-nuclear:
    "[OH][SbH][O][SbH][OH]": "distibonous acid",        # HO-SbH-O-SbH-OH
    "[OH][SbH][SbH][OH]": "hypodistibonous acid",       # (HO)HSb-SbH(OH)
    "[O]=[Sb]([OH])([OH])[O][Sb](=[O])([OH])[OH]": "distiboric acid",     # (HO)2Sb(O)-O-Sb(O)(OH)2
    "[O]=[Sb]([OH])([OH])[Sb](=[O])([OH])[OH]": "hypodistiboric acid",    # (HO)2(O)Sb-Sb(O)(OH)2
    "[OH][Sb]([OH])[O][Sb]([OH])[OH]": "distiborous acid",   # (HO)2Sb-O-Sb(OH)2
    "[OH][Sb]([OH])[Sb]([OH])[OH]": "hypodistiborous acid",  # (HO)2Sb-Sb(OH)2
    # Sulfur di-nuclear. disulfuric is above; dithionic/dithionous are
    # the S-S direct-bond 'hypo'-family preselected names.
    "O=S(=O)(O)S(=O)(=O)O": "dithionic acid",           # HO-SO2-SO2-OH (hypodisulfuric)
    "O=S(O)S(=O)O": "dithionous acid",                  # HO-SO-SO-OH (hypodisulfurous)
    # Tri-nuclear:
    "O=[PH](O)OP(=O)(O)O[PH](=O)O": "triphosphonic acid",       # (HO)HP(O)-O-HP(O)-O-HP(O)(OH)
    "O=P(O)(O)OP(=O)(O)OP(=O)(O)O": "triphosphoric acid",       # (HO)2P(O)-O-P(O)(OH)-O-P(O)(OH)2
    "O=S(=O)(O)OS(=O)(=O)OS(=O)(=O)O": "trisulfuric acid",      # HO-SO2-O-SO2-O-SO2-OH
}

# --- a phase: acid-halide + amide functional-class derivatives ---
# Acid halides of phosphoric/sulfuric (identical replaceable -OH groups) use the
# acyl-group word; amides replace all -OH by -NH2.
# These intercept @40 BEFORE the OPSIN-imported retained tier (RETAINED_NAME)
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
    "NP(N)(N)=O": "phosphoric triamide",   # PIN ('phosphoramide' = non-PIN alt)
    "NS(N)(=O)=O": "sulfuric diamide",     # PIN ('sulfamide' = general name)
    # Wave-2 completion C2 (carbonic-family composite-N parents; exact keys):
    "N=C(N)SSC(=N)N": "carbamimidic dithioperoxyanhydride",  # (formamidine disulfide; anhydride is the senior class per
    "NN=C(N)N": "carbonohydrazonic diamide",                 # (aminoguanidine, BB verbatim)
    "NN=C(NN)NN": "hydrazinecarbohydrazonohydrazide",        # (BB verbatim)
    "NS(=O)(=O)O": "sulfamic acid",        # H2N-SO2-OH (contraction of sulfuramidic)
    # Wave-2 P1AM (2026-07-09): hydrazine-parent carbonic-family parents.
    # (BB 32675): 'carboxamide' is ALWAYS the suffix on a
    # heteroacyclic parent -> hydrazinecarboxamide (PIN);
    # (BB 38623): "The systematic name is the preferred IUPAC name"
    # (semicarbazide = general nomenclature only). Intercepts @40 before
    # RETAINED_NAME which emitted 'semicarbazide'.
    "NNC(N)=O": "hydrazinecarboxamide",
    # (BB 34480 verbatim): amidrazone of carbonic acid.
    "N=C(NN)NN": "hydrazinecarboximidohydrazide",
    # family (aminoguanidine =NH tautomer; hydrazine parent +
    # carboximidamide suffix per; OPSIN-RT verified).
    "N=C(N)NN": "hydrazinecarboximidamide",
    # (BB 34298): "The names biguanide, triguanide, etc., are
    # no longer recommended. Condensed guanidines... are named systematically
    # as the diamides of imidodicarbonimidic acid". Bare parent only; the
    # @40 exact key never matches substituted forms, which keep the guanidine
    # handler's RT-valid general name until the N^n locant subsystem exists.
    "N=C(N)NC(=N)N": "imidodicarbonimidic diamide",
}

# --- a phase: carbonic/carbamic functional-replacement acids ---
# Plain (non-italic-locant) forms that OPSIN round-trips; names built by the
# shared FRN engine. The single-chalcogen tautomer forms needing italic S-/O-
# acid locants (carbonothioic S-acid) are DEFERRED to a phase (no OPSIN RT).
_CARBONIC_FRN = {
    "O=C(O)OO": build_frn_acid_name("carbon", "peroxo", 1),          # carbonoperoxoic acid
    "O=C(S)S": build_frn_acid_name("carbon", "thio", 2),             # carbonodithioic acid (HS-CO-SH)
    "S=C(S)S": build_frn_acid_name("carbon", "thio", 3),             # carbonotrithioic acid (HS-CS-SH)
    "N=C(O)O": build_frn_acid_name("carbon", "imido", 1),            # carbonimidic acid (HO-C(=NH)-OH)
    "N=C(N)O": build_frn_acid_name("carbam", "imido", 1),            # carbamimidic acid (H2N-C(=NH)-OH)
    "O=C(O)OC(=O)O": build_polyacid_name("carbonic acid", 2),        # dicarbonic acid
    "O=C(O)OC(=O)OC(=O)O": build_polyacid_name("carbonic acid", 3),  # tricarbonic acid
}

# --- W3-P05: carbonic-acid halides ---
# Carbonic acid (HO-CO-OH) with one -OH replaced by a halogen -> the retained
# ``carbono{halogen}idic acid`` PIN; BB 30684-30686: 'Cl-COOH
# carbonochloridic acid (PIN), not chloroformic acid'). (BB 30670)
# FORBIDS naming X-CO-OH as a substituted formic/methanoic acid, so the general
# chain-acid path's '1-chloro-1-oxomethanoic acid' is a non-PIN that MUST be
# intercepted @40. Exact full-molecule canonical-SMILES keys => zero false
# positives. Every name OPSIN-RT-confirmed.
_CARBONIC_ACID_HALIDES = {
    "O=C(O)Cl": "carbonochloridic acid",   # Cl-CO-OH
    "O=C(O)Br": "carbonobromidic acid",    # Br-CO-OH
    "O=C(O)F": "carbonofluoridic acid",    # F-CO-OH
    "O=C(O)I": "carbonoiodidic acid",      # I-CO-OH
}

# --- W3-P05: C-substituted formic acids (retained-name base) ---
# Formic acid (H-CO-OH) whose formyl H is replaced by an approved substituent is
# named on the retained 'formic acid' parent; BB 30692: 'permitted
# when substituent groups are other than those cited in '). This is NOT
# a general substituted-formic-acid engine forbids that: the
# halides/pseudohalides are their own retained 'carbono...idic acid' PINs, not
# 'chloroformic'/'cyanoformic'); each entry is a Blue-Book (PIN) whose substituent
# is off the forbidden list. Exact full-molecule canonical-SMILES key
# => zero false positives. Intercepts @40 before the general acid path; the nitro
# entry also preempts the (now-tightened) carbamic_acid SMARTS whose old form
# false-matched the nitro N and emitted a wrong 'carbamic acid'. OPSIN-RT-confirmed.
_C1_ACID_C_SUBSTITUTED = {
    "O=C(O)[N+](=O)[O-]": "nitroformic acid",              # O2N-CO-OH BB 30696
    # HOS2C-COOH: the -CS-O-SH acyl (dithiocarbonoperoxoyl) is off the
    # forbidden list, so the formic-parent name is a PIN (the Blue Book, location of
    # sulfur atoms unknown). The engine cannot build the compound acyl prefix, so
    # this is an exact-SMILES catalog entry (mirrors 'nitroformic acid').
    "O=C(O)C(=S)OS": "(dithiocarbonoperoxoyl)formic acid",  # the Blue Book
}

# --- W3-P05: carbonic-acid pseudohalides ---
# Carbonic acid (HO-CO-OH) with one -OH replaced by a pseudohalide (cyanido,
# azido) -> the retained ``carbono{pseudohalide}idic acid`` PIN
#, cyanido infix; BB 30832 'NC-CO-OH carbonocyanidic acid (PIN)'; BB 30834
# carbonazidic acid). As with the halides,  FORBIDS the
# substituted-formic-acid name ('1-cyanomethanoic acid'). Exact full-molecule
# canonical-SMILES keys => zero false positives: cyanoacetic acid (N#CCC(=O)O)
# has an extra CH2 and never matches, cascading to the general acid path. Every
# name OPSIN-RT-confirmed.
_CARBONIC_ACID_PSEUDOHALIDES = {
    "N#CC(=O)O": "carbonocyanidic acid",        # NC-CO-OH
    "[N-]=[N+]=NC(=O)O": "carbonazidic acid",   # N3-CO-OH
}

# --- W3-P15 /: hydroxylamine functional-parent
# PRESELECTED names ---
# Hydroxylamine (H2N-OH) is, exceptionally, a functional parent to which acid /
# amide suffixes attach at the OXYGEN atom (locant 'O'), and whose chalcogen
# analogues take the 'thio' functional-replacement prefix. These have retained /
# preselected PINs with no constitutional algorithm to run, so the correct PIN is
# an exact-canonical-SMILES table lookup (the model). Intercepts @40 before
# the general/skeletal paths that emitted 'unknown'. Exact full-molecule keys =>
# zero false positives (a substituted / charged derivative simply will not match
# and cascades onward). Every name OPSIN-RT-confirmed.
_HYDROXYLAMINE_PRESELECTED = {
    # (BB 38400): H2N-O-SO2-OH -> the -OH acid suffix sits on the
    # hydroxylamine O with locant 'O' ('not azanyl hydrogen sulfate').
    "NOS(=O)(=O)O": "hydroxylamine-O-sulfonic acid",
    # (BB 38446): H2N-SH -> chalcogen analogue of hydroxylamine
    # ('thio' functional-replacement); preselected name.
    "NS": "thiohydroxylamine",
}

# Merged lookup (no key overlap across the tables — distinct structures).
_ALL_INORGANIC = {
    **_INORGANIC_OXOACIDS,
    **_INORGANIC_ACID_DERIVATIVES,
    **_CARBONIC_FRN,
    **_CARBONIC_ACID_HALIDES,
    **_C1_ACID_C_SUBSTITUTED,
    **_CARBONIC_ACID_PSEUDOHALIDES,
    **_HYDROXYLAMINE_PRESELECTED,
}


def lookup_exact_acid_name(canonical_smiles: str) -> Optional[str]:
    """Return the retained / functional-replacement inorganic-acid PIN for an
    EXACT canonical-SMILES key (carbamimidic acid, carbonimidic acid,...), or
    None. Consumed by the ester path so a tabled acid analog keeps its
    retained '-ic acid' stem instead of the systematic-but-non-PIN chain name
    : 'carbamimidic acid', not '1-aminomethanimidic acid')."""
    return _ALL_INORGANIC.get(canonical_smiles)


def name_silicate_ester(mol) -> Optional[str]:
    """Tetraalkyl silicate ester PIN / BB 35978): a NEUTRAL silicon
    bearing exactly four ``-O-R`` groups (a fully-esterified silicic acid) ->
    ``{multiplier}{R} silicate`` (``tetramethyl silicate``,
    ``tetrakis(propan-2-yl) silicate``), or space-separated alphabetical citation
    for mixed R (``ethyl methyl... silicate``). The functional-class ester word
    is the PIN, NOT the substitutive ``tetra(R)oxysilane``.

    Fail-closed (returns ``None``) for ANY Si that is not exactly Si(OR)4: a free
    Si-OH (partial ester / silicic acid -> stays the free-acid table row or the
    substitutive silanetriol path), an Si-C bond (genuine organosilicon ->
     / mononuclear-hydride), a charge, or a non-carbon-rooted O substituent.
    """
    if mol is None:
        return None
    si_atoms = [a for a in atoms_of(mol) if a.GetSymbol() == "Si"]
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
        alpha_sort_key,
        get_multiplier_prefix,
        is_complex_substituent,
    )
    counts = Counter(r_names)
    parts = []
    for nm in sorted(counts, key=alpha_sort_key):
        c = counts[nm]
        enclosed = f"({nm})" if is_complex_substituent(nm) else nm
        from ..assembly.naming_utils import multiplied_component as _mc
        parts.append(enclosed if c == 1 else _mc(c, nm, enclosed))
    return " ".join(parts) + " silicate"


def _is_cyanatido_oxygen(mol, o_atom, p_idx: int) -> bool:
    """True iff ``o_atom`` (a single-bonded neighbour of the acid P) is the O of an
    ``-O-C#N`` cyanatido group: O bonded only to P and a two-
    coordinate carbon that is triple-bonded to a terminal N."""
    others = [x for x in o_atom.GetNeighbors() if x.GetIdx() != p_idx]
    if len(others) != 1 or others[0].GetSymbol() != "C":
        return False
    c = others[0]
    if c.GetDegree() != 2:
        return False
    c_others = [x for x in c.GetNeighbors() if x.GetIdx() != o_atom.GetIdx()]
    if len(c_others) != 1 or c_others[0].GetSymbol() != "N":
        return False
    n = c_others[0]
    if n.GetDegree() != 1 or n.GetFormalCharge() != 0:
        return False
    return mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondType() == \
        Chem.BondType.TRIPLE


def _amido_n_prefix(mol, n_idx: int, p_idx: int) -> Optional[str]:
    """Return the N-locant detachable-prefix string for an amido nitrogen
    ``-N(R)(R')`` bonded to the acid P — ``"N,N-dimethyl"``, ``"N-methyl"``,
    ``"N-ethyl-N-methyl"`` — ``""`` for a bare ``-NH2``, or ``None`` (fail-closed)
    when any N-substituent is not a simple unbranched alkyl / phenyl-naphthyl aryl
    : nonacidic-hydrogen substitution cited with the italic N locant).
    """
    from collections import Counter

    from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound, multiplier_needs_hyphen
    from .substituent_purity import organyl_prefix_name

    n = mol.GetAtomWithIdx(n_idx)
    sub_names: list = []
    for nb in n.GetNeighbors():
        if nb.GetIdx() == p_idx:
            continue
        if mol.GetBondBetweenAtoms(n_idx, nb.GetIdx()).GetBondType() != \
                Chem.BondType.SINGLE:
            return None
        nm = organyl_prefix_name(mol, nb.GetIdx(), n_idx)
        if nm is None:
            return None
        sub_names.append(nm)
    if not sub_names:
        return ""                                    # bare -NH2
    _MULT = {1: "", 2: "di", 3: "tri"}
    counts = Counter(sub_names)
    segs = []
    #: the guard above is the shared chokepoint, so a prefix here may carry
    # a locant or a retained italicized prefix. The italic-N locant is joined by a
    # hyphen, so an unmarked locanted prefix would read as two locant sets --
    # enclosing marks come from the shared primitive:
    # BB 32784 `N-(propan-2-yl)acetamide` (PIN) -> locanted organyl enclosed;
    # BB 3465 `4-butyl-4-tert-butylcyclohexan-1-ol` (PIN) -> the retained
    # italicized prefix is cited BARE even straight after a locant
    # (the BB 3465 example itself; the rule is (d) for the
    # hyphen and (b)/BB 7070 for the simple multiplier, NOT
    #, which is the parentheses rule). Hence the bare
    # N-substituent form, and `N,N-di-tert-butyl` keeps the
    # multiplier's hyphen. ('N-tert-butyl' is not itself a BB example.)
    for nm in sorted(counts, key=alpha_sort_key):
        c = counts[nm]
        if c not in _MULT:
            return None
        locs = ",".join(["N"] * c)
        marked = enclose_if_compound(nm)
        if _MULT[c] and marked == nm and multiplier_needs_hyphen(nm):
            segs.append(f"{locs}-{_MULT[c]}-{marked}")
        else:
            segs.append(f"{locs}-{_MULT[c]}{marked}")
    return "-".join(segs)


def name_p_oxoacid_frn(mol) -> Optional[str]:
    """ mononuclear P/As/Sb oxoacid functional-replacement PIN (Engine A).

    A NEUTRAL, single-fragment molecule with exactly one central atom E in
    {P, As, Sb} that bears one ``E=O``, at least one remaining ``-OH`` (so the
    class is 'acid',, and at least one ``-OH`` REPLACED by a class
    group — amido (``-N(R)(R')``), halido (``-Cl/-F/-Br/-I``) or cyanatido
    (``-O-C#N``). The skeletal (C + H) substituent count on E selects the parent:
    0 -> the -oric stem (``phosphor`` / ``arsor`` / ``stibor``); 1 -> the -onic
    stem (``phosphon`` / ``arson`` / ``stibon``).: a C-E bond forces
    the '-ono' stem, never '-ini'; a halogen/N/replaced-O is a CLASS infix, never
    skeletal, so ``Cl-E(O)(OH)2`` is ``E-orochloridic acid``, never
    ``chloro-E-onic acid``.

        (CH3)2N-P(O)(OH)2 -> N,N-dimethylphosphoramidic acid
        CH3-P(O)(OCN)(OH) -> methylphosphonocyanatidic acid
        C6H5-P(O)(Cl)(OH) -> phenylphosphonochloridic acid
        Cl-As(O)(OH)2 -> arsorochloridic acid (guard,
        CH3-As(O)(Cl)(OH) -> methylarsonochloridic acid

    Fail-closed (returns ``None``) off this exact shape: a plain -onic / -oric acid
    (no class group -> the existing suffix/table paths own it), a fully-replaced
    acid with no -OH (an amide / acid-halide functional class), any ester ``-OR``
    or chalcogenol ``-SH/-SeH/-TeH`` neighbour (chalcogen-infix FRN is a separate
    path), an E=N/E#N/E=S multiple bond, >=2 organyl groups (-inic FRN — out of
    scope), a molecule carrying more than one P/As/Sb, or the organyl+N-substituent
    combination that needs E-/N- locant disambiguation. Pure: no mol mutation.
    """
    from ..assembly.naming_utils import enclose_if_compound
    from .substituent_purity import organyl_prefix_name

    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    # a phase: central atom is any one of P / As / Sb (the guard table keys),
    # exactly one of them and no other pnictogen present (a molecule with both a P
    # and an As is not a mononuclear oxoacid).
    ctrs = [a for a in atoms_of(mol) if a.GetSymbol() in _FRN_CENTRAL_STEMS]
    if len(ctrs) != 1:
        return None
    p = ctrs[0]
    symbol = p.GetSymbol()
    if p.GetFormalCharge() != 0 or p.GetNumRadicalElectrons() != 0:
        return None

    dbl_o = 0
    single_nbrs = []
    for b in p.GetBonds():
        o = b.GetOtherAtom(p)
        bt = b.GetBondType()
        if bt == Chem.BondType.DOUBLE and o.GetSymbol() == "O":
            dbl_o += 1
        elif bt == Chem.BondType.SINGLE:
            single_nbrs.append(o)
        else:
            return None                              # P=N / P#N / P=S -> not a target
    if dbl_o != 1:
        return None

    oh_count = 0
    organyl_c: list = []
    amido_n: list = []
    infix_counts: dict = {}
    for nb in single_nbrs:
        sym = nb.GetSymbol()
        if sym == "O":
            if nb.GetFormalCharge() != 0:
                return None
            others = [x for x in nb.GetNeighbors() if x.GetIdx() != p.GetIdx()]
            if nb.GetTotalNumHs() == 1 and not others:
                oh_count += 1
            elif _is_cyanatido_oxygen(mol, nb, p.GetIdx()):
                infix_counts["cyanatido"] = infix_counts.get("cyanatido", 0) + 1
            else:
                return None                          # ester -OR / peroxide -> defer
        elif sym == "C":
            organyl_c.append(nb.GetIdx())
        elif sym in _HALIDO_INFIX:
            k = _HALIDO_INFIX[sym]
            infix_counts[k] = infix_counts.get(k, 0) + 1
        elif sym == "N":
            if nb.GetFormalCharge() != 0:
                return None
            if any(b.GetBondType() != Chem.BondType.SINGLE for b in nb.GetBonds()):
                return None                          # imido/hydrazono/nitrido -> defer
            amido_n.append(nb.GetIdx())
            infix_counts["amido"] = infix_counts.get("amido", 0) + 1
        else:
            return None

    if oh_count < 1 or not infix_counts:             # class 'acid' + >=1 replacement
        return None
    skeletal = len(organyl_c) + p.GetTotalNumHs()    #: only C + H
    parent_stem = _FRN_CENTRAL_STEMS[symbol].get(skeletal)
    if parent_stem is None:
        return None                                  # -inic FRN (skeletal 2) out of scope

    organyl_names = []
    for c_idx in organyl_c:
        nm = organyl_prefix_name(mol, c_idx, p.GetIdx())
        if nm is None:
            return None
        organyl_names.append(nm)
    n_prefixes = []
    for n_idx in amido_n:
        seg = _amido_n_prefix(mol, n_idx, p.GetIdx())
        if seg is None:
            return None
        if seg:
            n_prefixes.append(seg)
    if organyl_names and n_prefixes:
        return None                                  # needs P-/N- locants -> defer

    if organyl_names:
        # `build_p_frn_acid_name` concatenates `front` straight onto the stem and
        # does no marking of its own, so a compound prefix takes its marks
        # here or it runs into the stem ('propan-2-ylphosphono...'). The shared
        # primitive keeps every simple prefix BARE, as `phenylphosphonochloridic
        # acid` already was.
        front = enclose_if_compound(organyl_names[0])
    elif n_prefixes:
        front = "-".join(sorted(n_prefixes))
    else:
        front = ""
    return build_p_frn_acid_name(front, parent_stem, infix_counts)


# a phase: single-bonded halogen -> halide class word.
_HALIDE_CLASS_WORD = {"Cl": "chloride", "F": "fluoride", "Br": "bromide",
                      "I": "iodide"}


def _halide_pseudohalide_class_word(mol, nb, ctr_idx: int) -> Optional[str]:
    """ class word for a halide / pseudohalide attached to the central
    atom, else ``None`` (so an unrecognised neighbour fails the caller closed).

    Detection is tight (0-wrong): a bare halogen; azide ``-N=[N+]=[N-]``;
    isocyanate ``-N=C=O`` / isothiocyanate ``-N=C=S``; cyanide ``-C#N``. Anything
    else (an amido -N<, an -OH, an ester -OR, an organyl) returns ``None`` and is
    handled by the caller's own classification.
    """
    sym = nb.GetSymbol()
    if sym in _HALIDE_CLASS_WORD:
        b = mol.GetBondBetweenAtoms(ctr_idx, nb.GetIdx())
        if b.GetBondType() == Chem.BondType.SINGLE and nb.GetDegree() == 1:
            return _HALIDE_CLASS_WORD[sym]
        return None
    others = [x for x in nb.GetNeighbors() if x.GetIdx() != ctr_idx]
    # cyanide ctr-C#N
    if sym == "C" and nb.GetTotalNumHs() == 0 and len(others) == 1:
        o = others[0]
        bond = mol.GetBondBetweenAtoms(nb.GetIdx(), o.GetIdx())
        if o.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.TRIPLE:
            return "cyanide"
        return None
    if sym != "N" or nb.GetTotalNumHs() != 0 or len(others) != 1:
        return None
    o = others[0]
    ctr_bond = mol.GetBondBetweenAtoms(ctr_idx, nb.GetIdx())
    if ctr_bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    mid_bond = mol.GetBondBetweenAtoms(nb.GetIdx(), o.GetIdx())
    # azide ctr-N=[N+]=[N-]
    if o.GetSymbol() == "N" and mid_bond.GetBondType() == Chem.BondType.DOUBLE:
        tails = [x for x in o.GetNeighbors() if x.GetIdx() != nb.GetIdx()]
        if (len(tails) == 1 and tails[0].GetSymbol() == "N"
                and mol.GetBondBetweenAtoms(o.GetIdx(), tails[0].GetIdx()).GetBondType()
                == Chem.BondType.DOUBLE and tails[0].GetDegree() == 1):
            return "azide"
        return None
    # isocyanate ctr-N=C=O / isothiocyanate ctr-N=C=S (cumulated double bonds)
    if o.GetSymbol() == "C" and mid_bond.GetBondType() == Chem.BondType.DOUBLE:
        cts = [x for x in o.GetNeighbors() if x.GetIdx() != nb.GetIdx()]
        if (len(cts) == 1
                and mol.GetBondBetweenAtoms(o.GetIdx(), cts[0].GetIdx()).GetBondType()
                == Chem.BondType.DOUBLE and cts[0].GetDegree() == 1):
            if cts[0].GetSymbol() == "O":
                return "isocyanate"
            if cts[0].GetSymbol() == "S":
                return "isothiocyanate"
    return None


# a phase: chalcogenol -> chalcogen infix /.
_CHALCOGEN_INFIX = {"S": "thio", "Se": "seleno", "Te": "telluro"}


def name_p_oxoacid_chalcogen_frn(mol) -> Optional[str]:
    """ mononuclear P/As/Sb oxoacid CHALCOGEN-infix FRN (locant-free
    subset only). E(=O) with two or more acidic -OH replaced by the SAME chalcogen
    -SH / -SeH / -TeH, giving the position-undetermined, locant-free name::

        As(O)(OH)(SH)2 -> arsorodithioic acid (BB L35636)
        P(O)(SH)3 -> phosphorotrithioic acid
        C6H5-P(O)(SH)2 -> phenylphosphonodithioic acid

    SAFELY NARROW by design (0-wrong + the tautomer-locant spelling-blind
    spot): it fires ONLY when the =O is present and >=2 replacements are the SAME
    chalcogen, so a locant is provably omitted. It FAILS CLOSED on every case that
    needs an italic tautomer locant — a MONO replacement (``arsorothioic S-acid`` /
    ``O,O,O-acid``), a =S/=Se double bond, a mixed chalcogen set, or a peroxo /
    class-infix mix. Those belong to a phase Group A's derivation; here
    they degrade to the existing paths / abstain, never to a wrong spelling. Every
    emission is OPSIN-round-trip gated downstream. Pure: no mol mutation.
    """
    from .functional_replacement import build_frn_acid_name
    from ..assembly.naming_utils import enclose_if_compound
    from .substituent_purity import organyl_prefix_name

    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    ctrs = [a for a in atoms_of(mol) if a.GetSymbol() in _FRN_CENTRAL_STEMS]
    if len(ctrs) != 1:
        return None
    ctr = ctrs[0]
    symbol = ctr.GetSymbol()
    if ctr.GetFormalCharge() != 0 or ctr.GetNumRadicalElectrons() != 0:
        return None

    # -oric/-onic bare stems WITHOUT the linking vowel (build_frn_acid_name adds it).
    _FRN_BARE = {"P": {0: "phosphor", 1: "phosphon"},
                 "As": {0: "arsor", 1: "arson"},
                 "Sb": {0: "stibor", 1: "stibon"}}
    dbl_o = 0
    organyl_c: list = []
    chalco_counts: dict = {}
    oh_count = 0
    for b in ctr.GetBonds():
        o = b.GetOtherAtom(ctr)
        bt = b.GetBondType()
        if bt == Chem.BondType.DOUBLE and o.GetSymbol() == "O":
            dbl_o += 1
            continue
        if bt != Chem.BondType.SINGLE:
            return None                          # =S/=Se/=N -> tautomer-locant, defer
        sym = o.GetSymbol()
        if sym == "O":
            if o.GetFormalCharge() != 0:
                return None
            others = [x for x in o.GetNeighbors() if x.GetIdx() != ctr.GetIdx()]
            if o.GetTotalNumHs() == 1 and not others:
                oh_count += 1
            else:
                return None                      # ester -OR / peroxide -> defer
        elif sym in _CHALCOGEN_INFIX:
            others = [x for x in o.GetNeighbors() if x.GetIdx() != ctr.GetIdx()]
            if o.GetFormalCharge() != 0 or o.GetTotalNumHs() != 1 or others:
                return None                      # -S-R / -S-S- etc. -> defer
            chalco_counts[sym] = chalco_counts.get(sym, 0) + 1
        elif sym == "C":
            organyl_c.append(o.GetIdx())
        else:
            return None                          # halogen/N -> not a plain chalcogen FRN
    if dbl_o != 1:
        return None
    if len(chalco_counts) != 1:                  # exactly one chalcogen type
        return None
    chalco_sym, n = next(iter(chalco_counts.items()))
    if n < 2:                                    # MONO -> needs a tautomer locant, defer
        return None
    skeletal = len(organyl_c) + ctr.GetTotalNumHs()
    base = _FRN_BARE[symbol].get(skeletal)
    if base is None:                             # -inic (skeletal 2) can't hold di+, defer
        return None

    front = ""
    if organyl_c:
        nm = organyl_prefix_name(mol, organyl_c[0], ctr.GetIdx())
        if nm is None:
            return None
        front = enclose_if_compound(nm)
    acid = build_frn_acid_name(base, _CHALCOGEN_INFIX[chalco_sym], n)
    if acid is None:
        return None
    return f"{front}{acid}"


def name_p_oxoacid_halide_amide(mol) -> Optional[str]:
    """ / mononuclear P/As/Sb oxoacid HALIDE / AMIDE PIN.

    The ``oh_count == 0`` sibling of:func:`name_p_oxoacid_frn`: every acidic -OH
    is replaced, so the compound is not class 'acid' — its class word is a halide /
    pseudohalide or, when no halide is present and every -OH became
    -NH2, an amide. The acid STEM WORD is the same one
    :func:`name_p_oxoacid_frn` builds (``phenylphosphonous``, ``phenylphosphonic``,
    ``N,N-dimethylphosphoramidic``); the halide/amide builders append the class
    word(s) in / (b) seniority order::

        C6H5-PBrCl -> phenylphosphonous bromide chloride (BB L35718)
        (C6H5)2P-Cl -> diphenylphosphinous chloride (BB L35712)
        C6H5-PCl2 -> phenylphosphonous dichloride (BB L35716)
        (CH3)2N-P(O)Cl2 -> N,N-dimethylphosphoramidic dichloride

    Fail-closed (``None``) off this exact shape. In particular it declines the
    bare skeletal-0 no-infix case (``POCl3`` = phosphoryl trichloride, the
     acyl-word exception — owned by the existing acyl path), any -OH
    left (that is class 'acid' -> the acid paths own it), an E=N/E=S multiple bond,
    an ester -OR neighbour, and the organyl+N-substituent combination that needs
    E-/N- locant disambiguation. Pure: no mol mutation.
    """
    from ..rules.phosphorus import (_PNICTOGEN_OUS_STEMS, _PNICTOGEN_OXOACID_STEMS,
                                    _build_substituent_string)
    from .functional_replacement import (build_p_frn_acid_stem_word,
                                          build_p_frn_amide_name,
                                          build_p_frn_halide_name)
    from ..assembly.naming_utils import enclose_if_compound
    from .substituent_purity import organyl_prefix_name

    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    ctrs = [a for a in atoms_of(mol) if a.GetSymbol() in _FRN_CENTRAL_STEMS]
    if len(ctrs) != 1:
        return None
    ctr = ctrs[0]
    symbol = ctr.GetSymbol()
    if ctr.GetFormalCharge() != 0 or ctr.GetNumRadicalElectrons() != 0:
        return None

    oxo = 0
    organyl_c: list = []
    amido_n: list = []
    class_counts: dict = {}          # halide/pseudohalide class word -> count
    amide_n: list = []               # bare/substituted -NH2 (amide class candidates)
    for b in ctr.GetBonds():
        o = b.GetOtherAtom(ctr)
        bt = b.GetBondType()
        if bt == Chem.BondType.DOUBLE and o.GetSymbol() == "O":
            oxo += 1
            continue
        if bt != Chem.BondType.SINGLE:
            return None                              # E=N / E=S / E#N -> defer
        cls = _halide_pseudohalide_class_word(mol, o, ctr.GetIdx())
        if cls is not None:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            continue
        sym = o.GetSymbol()
        if sym == "O":
            if o.GetFormalCharge() != 0:
                return None
            nb_others = [x for x in o.GetNeighbors() if x.GetIdx() != ctr.GetIdx()]
            if o.GetTotalNumHs() == 1 and not nb_others:
                return None                          # -OH left -> class 'acid', defer
            return None                              # ester -OR / other O -> defer
        if sym == "C":
            organyl_c.append(o.GetIdx())
        elif sym == "N":
            if o.GetFormalCharge() != 0:
                return None
            if any(bd.GetBondType() != Chem.BondType.SINGLE for bd in o.GetBonds()):
                return None                          # imido/hydrazono -> defer
            amido_n.append(o.GetIdx())
            amide_n.append(o.GetIdx())
        else:
            return None
    if oxo not in (0, 1):
        return None
    skeletal = len(organyl_c) + ctr.GetTotalNumHs()

    organyl_names = []
    for c_idx in organyl_c:
        nm = organyl_prefix_name(mol, c_idx, ctr.GetIdx())
        if nm is None:
            return None
        organyl_names.append(nm)

    # ---- principal class = halide/pseudohalide if present, else amide -----------
    if class_counts:
        # halide/pseudohalide derivative: any -N< becomes an amido INFIX.
        n_prefixes = []
        for n_idx in amido_n:
            seg = _amido_n_prefix(mol, n_idx, ctr.GetIdx())
            if seg is None:
                return None
            if seg:
                n_prefixes.append(seg)
        if organyl_names and n_prefixes:
            return None                              # E-/N- locant disambiguation, defer
        infix_counts = {"amido": len(amido_n)} if amido_n else {}
        stem_word = _pnictogen_acid_stem_word(
            symbol, oxo, skeletal, organyl_names, n_prefixes, infix_counts,
            _PNICTOGEN_OXOACID_STEMS, _PNICTOGEN_OUS_STEMS,
            build_p_frn_acid_stem_word, enclose_if_compound, _build_substituent_string)
        if stem_word is None:
            return None
        return build_p_frn_halide_name(stem_word, class_counts)

    # ---- amide/hydrazide derivative (no halide; every -OH -> -NH2) --------------
    if amide_n and not amido_n_has_substituent(mol, amide_n, ctr.GetIdx()):
        # principle: the amide of a *bare* mononuclear hydride whose
        # parent hydride is a preselected name is named SUBSTITUTIVELY, not by the
        # functional class 'amide' (H2B-NH2 -> boranamine, *not* borinic amide).
        # Phosphane/arsane/stibane (PH3/AsH3/SbH3) are preselected parent hydrides
        #, so the bare trivalent -ous hydride amide H2E-NH2 / HE(NH2)2
        # (oxo == 0 AND no organyl) is the substitutive amine phosphanamine /
        # arsanamine / stibanamine, amine is expressed as suffix), NOT
        # 'phosphinous amide'. The functional-class -ous amide stays the PIN only
        # when the acid carries organyl substituents (cf. dimethylphosphinous
        # hydrazide, BB L23150; phenylphosphonous diamide) or is an oxoacid
        # (oxo >= 1). Decline here so the substitutive namer builds the PIN.
        if oxo == 0 and not organyl_names:
            return None
        stem_word = _pnictogen_acid_stem_word(
            symbol, oxo, skeletal, organyl_names, [], {},
            _PNICTOGEN_OXOACID_STEMS, _PNICTOGEN_OUS_STEMS,
            build_p_frn_acid_stem_word, enclose_if_compound, _build_substituent_string)
        if stem_word is None:
            return None
        return build_p_frn_amide_name(stem_word, "amide", len(amide_n))
    return None


def amido_n_has_substituent(mol, n_idxs, ctr_idx: int) -> bool:
    """True if any candidate amide nitrogen carries a non-H substituent (an
    N-substituted amide needs N-/P- locant assembly — out of this simple scope)."""
    for n_idx in n_idxs:
        n = mol.GetAtomWithIdx(n_idx)
        if any(x.GetIdx() != ctr_idx for x in n.GetNeighbors()):
            return True
    return False


def _pnictogen_acid_stem_word(symbol, oxo, skeletal, organyl_names, n_prefixes,
                              infix_counts, oxo_stems, ous_stems,
                              build_stem_word, enclose_if_compound,
                              build_sub_string) -> Optional[str]:
    """Return the acid name minus ``" acid"`` for a P/As/Sb halide/amide parent,
    else ``None``. Reused by:func:`name_p_oxoacid_halide_amide`."""
    if infix_counts:
        # class-infix FRN parent (amido) -> -oric/-onic stem, pentavalent only.
        if not oxo:
            return None
        stem_base = _FRN_CENTRAL_STEMS[symbol].get(skeletal)
        if stem_base is None:
            return None
        if organyl_names:
            front = enclose_if_compound(organyl_names[0])
        elif n_prefixes:
            front = "-".join(sorted(n_prefixes))
        else:
            front = ""
        return build_stem_word(front, stem_base, infix_counts)
    # plain parent (no class infix): -onic/-inic (oxo) or -onous/-inous (trivalent).
    table = oxo_stems if oxo else ous_stems
    if skeletal == 1:
        word = table[symbol][0]
    elif skeletal == 2:
        word = table[symbol][1]
    else:
        return None            # skeletal 0 plain -> phosphoryl/acyl-word exception, defer
    front = build_sub_string(organyl_names) if organyl_names else ""
    return f"{front}{word}"


def name_p_frn_ester(mol) -> Optional[str]:
    """ FRN-modified phosphorus-acid ESTER PIN.

    An ester of a functional-replacement-modified phosphorus acid: the central P
    carries one or more ``-O-R`` ester owners AND at least one class infix (amido
    ``-N<`` / halido ``-Cl/-F/-Br/-I`` / cyanatido ``-OCN``). The ester suffix is
    ``-ate`` for the pentavalent (``P=O``) acid and ``-ite`` for the trivalent one,
    replacing the acid's ``-ic`` ending on the SAME stem word that
    :func:`build_p_frn_acid_stem_word` builds::

        CH3-O-P(Cl)-N(CH3)2 -> methyl N,N-dimethylphosphoramidochloridite
        CH3-O-P(=O)(Cl)-NH2 -> methyl phosphoramidochloridate

    Requires >=1 ester owner AND >=1 class infix, so a plain phosphate/phosphite
    ester (owned by:func:`~orthonym.rules.phosphorus.name_phosphate_ester`) and a
    free FRN acid (owned by:func:`name_p_oxoacid_frn`) never reach here — this is a
    non-overlapping ADDITION, not a change to the hot ester path. Fail-closed off
    the shape (an ester -OR to non-C/S, an E=N/E=S bond, the organyl+N-substituent
    combination needing E-/N- locants, or an atom left unaccounted). Pure: no mol
    mutation.
    """
    from ..rules.phosphorus import (_p_ester_owner_group, _assemble_p_owner_text,
                                    _HYDROGEN_MULT)
    from .functional_replacement import build_p_frn_acid_stem_word
    from ..assembly.naming_utils import enclose_if_compound
    from .substituent_purity import organyl_prefix_name

    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    ps = [a for a in atoms_of(mol) if a.GetSymbol() == "P"]   # P esters only (as name_phosphate_ester)
    if len(ps) != 1:
        return None
    p = ps[0]
    if p.GetFormalCharge() != 0 or p.GetNumRadicalElectrons() != 0 or p.IsInRing():
        return None

    accounted = {p.GetIdx()}
    dbl_o = 0
    oh_count = 0
    organyl_c: list = []
    amido_n: list = []
    ester_oxys: list = []
    infix_counts: dict = {}
    for b in p.GetBonds():
        o = b.GetOtherAtom(p)
        bt = b.GetBondType()
        if bt == Chem.BondType.DOUBLE and o.GetSymbol() == "O":
            dbl_o += 1
            accounted.add(o.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return None                              # P=N/P=S/P#N -> defer
        sym = o.GetSymbol()
        if sym == "O":
            if o.GetFormalCharge() != 0:
                return None
            others = [x for x in o.GetNeighbors() if x.GetIdx() != p.GetIdx()]
            if not others and o.GetTotalNumHs() >= 1:
                oh_count += 1
                accounted.add(o.GetIdx())
            elif len(others) == 1 and others[0].GetSymbol() in ("C", "S"):
                ester_oxys.append(o.GetIdx())        # -O-R ester owner
                accounted.add(o.GetIdx())
            elif _is_cyanatido_oxygen(mol, o, p.GetIdx()):
                infix_counts["cyanatido"] = infix_counts.get("cyanatido", 0) + 1
                accounted.add(o.GetIdx())
                for x in o.GetNeighbors():           # the O-C#N atoms
                    if x.GetIdx() != p.GetIdx():
                        accounted.add(x.GetIdx())
                        for y in x.GetNeighbors():
                            accounted.add(y.GetIdx())
            else:
                return None                          # P-O-P / O-N -> defer
        elif sym == "C":
            organyl_c.append(o.GetIdx())
        elif sym in _HALIDO_INFIX:
            k = _HALIDO_INFIX[sym]
            infix_counts[k] = infix_counts.get(k, 0) + 1
            accounted.add(o.GetIdx())
        elif sym == "N":
            if o.GetFormalCharge() != 0:
                return None
            if any(bd.GetBondType() != Chem.BondType.SINGLE for bd in o.GetBonds()):
                return None                          # imido/hydrazono -> defer
            amido_n.append(o.GetIdx())
            infix_counts["amido"] = infix_counts.get("amido", 0) + 1
        else:
            return None

    if not ester_oxys:
        return None                                  # no ester owner -> not this class
    # require >=1 CLASS infix (else it's a plain ester -> name_phosphate_ester owns it)
    if not infix_counts:
        return None
    if dbl_o > 1:
        return None

    skeletal = len(organyl_c) + p.GetTotalNumHs()
    parent_stem = _FRN_CENTRAL_STEMS["P"].get(skeletal)
    if parent_stem is None:
        return None                                  # -inic FRN ester out of scope

    organyl_names = []
    for c_idx in organyl_c:
        nm = organyl_prefix_name(mol, c_idx, p.GetIdx())
        if nm is None:
            return None
        organyl_names.append(nm)
    n_prefixes = []
    for n_idx in amido_n:
        seg = _amido_n_prefix(mol, n_idx, p.GetIdx())
        if seg is None:
            return None
        if seg:
            n_prefixes.append(seg)
        accounted.add(n_idx)
        for x in mol.GetAtomWithIdx(n_idx).GetNeighbors():   # N-substituent carbons
            if x.GetIdx() != p.GetIdx():
                stack = [x.GetIdx()]
                while stack:
                    a = stack.pop()
                    if a in accounted:
                        continue
                    accounted.add(a)
                    for y in mol.GetAtomWithIdx(a).GetNeighbors():
                        if y.GetIdx() != n_idx and y.GetIdx() not in accounted:
                            stack.append(y.GetIdx())
    if organyl_names and n_prefixes:
        return None                                  # P-/N- locant disambiguation -> defer

    if organyl_names:
        front = enclose_if_compound(organyl_names[0])
    elif n_prefixes:
        front = "-".join(sorted(n_prefixes))
    else:
        front = ""

    stem_word = build_p_frn_acid_stem_word(front, parent_stem, infix_counts)
    if stem_word is None or not stem_word.endswith("ic"):
        return None
    # ester suffix: -ate for the pentavalent (P=O) acid, -ite for the trivalent.
    ester_stem = stem_word[:-2] + ("ate" if dbl_o == 1 else "ite")

    owner_tokens = []
    for o_idx in ester_oxys:
        got = _p_ester_owner_group(mol, o_idx, p.GetIdx())
        if got is None:
            return None
        token, frag = got
        owner_tokens.append(token)
        accounted |= frag
    owner_text = _assemble_p_owner_text(owner_tokens)

    if accounted != set(range(mol.GetNumAtoms())):
        return None                                  # unaccounted atom -> not whole-molecule

    hyd = _HYDROGEN_MULT.get(oh_count)
    if hyd is None:
        return None
    pieces = [owner_text]
    if hyd:
        pieces.append(hyd)
    pieces.append(ester_stem)
    return " ".join(pieces)


# Chalcogen replacement -> prefix /. No linking-o
# elision on these prefixes 'thiosilicic acid').
_CHALCOGENOL_PREFIX = {"S": "thio", "Se": "seleno", "Te": "telluro"}
_CHALCOGEN_PREFIX_ORDER = {"thio": 2, "seleno": 0, "telluro": 1}  # alpha: seleno<telluro<thio


def name_silicic_acid_frn(mol) -> Optional[str]:
    """ chalcogen-prefix functional replacement on silicic acid.

    Silicic acid Si(OH)4 with one or more -OH replaced by -SH / -SeH / -TeH ->
    ``{thio|seleno|telluro}silicic acid`` (prefixes, cited alphabetically, no 'o'
    elision, no italic locant; BB L35670 ``Si(OH)3(SH) thiosilicic acid``).

    Fail-closed (returns ``None``) off this exact shape: any Si-C bond (genuine
    organosilicon), any non-``OH``/non-chalcogenol neighbour, no chalcogen
    replacement at all (-> the tabled ``silicic acid`` row owns it), no -OH left
    (not class 'acid'), a charge, or more than one Si. Pure: no mol mutation.
    """
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    sis = [a for a in atoms_of(mol) if a.GetSymbol() == "Si"]
    if len(sis) != 1:
        return None
    si = sis[0]
    if si.GetFormalCharge() != 0 or si.GetDegree() != 4 or si.GetTotalNumHs() != 0:
        return None
    oh_count = 0
    chalco: dict = {}
    for b in si.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            return None
        nb = b.GetOtherAtom(si)
        sym = nb.GetSymbol()
        if (sym == "O" and nb.GetFormalCharge() == 0 and nb.GetDegree() == 1
                and nb.GetTotalNumHs() == 1):
            oh_count += 1
        elif (sym in _CHALCOGENOL_PREFIX and nb.GetFormalCharge() == 0
                and nb.GetDegree() == 1 and nb.GetTotalNumHs() == 1):
            pre = _CHALCOGENOL_PREFIX[sym]
            chalco[pre] = chalco.get(pre, 0) + 1
        else:
            return None
    if oh_count < 1 or not chalco:
        return None
    _MULT = {1: "", 2: "di", 3: "tri"}
    parts = []
    for pre in sorted(chalco, key=lambda p: _CHALCOGEN_PREFIX_ORDER[p]):
        c = chalco[pre]
        if c not in _MULT:
            return None
        parts.append(f"{_MULT[c]}{pre}")
    return "".join(parts) + "silicic acid"


def name_borane_amine(mol) -> Optional[str]:
    """: amides of the boron acids are named SUBSTITUTIVELY on the
    parent hydride borane (BH3), not as ``boric triamide``.

        B(NH2)3 -> boranetriamine (BB L35838; 'not boric triamide')
        HB(NH2)2 -> boranediamine
        H2B-NH2 -> boranamine

    Fail-closed (returns ``None``) off this exact shape: any B-C or B-O bond (a
    carbon-bearing boron -> boronic path; an oxygen-bearing boron -> the boric-acid
    table), an N that is not a terminal ``-NH2``, a charge, more than one B, or any
    heavy atom other than B/N. Pure: no mol mutation.
    """
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    bs = [a for a in atoms_of(mol) if a.GetSymbol() == "B"]
    if len(bs) != 1:
        return None
    b = bs[0]
    if b.GetFormalCharge() != 0 or b.GetNumRadicalElectrons() != 0:
        return None
    # No heavy atom other than the single B and its amine nitrogens.
    if any(a.GetSymbol() not in ("B", "N", "H") for a in atoms_of(mol)):
        return None
    n_count = 0
    for bond in b.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        nb = bond.GetOtherAtom(b)
        if (nb.GetSymbol() != "N" or nb.GetFormalCharge() != 0
                or nb.GetDegree() != 1 or nb.GetTotalNumHs() != 2):
            return None                              # only terminal -NH2
        n_count += 1
    if n_count != b.GetDegree() or n_count < 1:
        return None
    if n_count == 1:
        return "boranamine"                          # borane + amine (e elided)
    return "borane" + {2: "di", 3: "tri", 4: "tetra"}.get(n_count, "") + "amine"


# --- W3-P10 Engine B: di-/polynuclear noncarbon-oxoacid derivatives ---
# Parent (preselected) names keyed by (acid-centre element, bridge slot). A direct
# centre-centre bond (no bridge) is the 'hypo' parent.
#
# a phase B.2 /.2.4): the DERIVATIVE element set is extended from
# {P, S, Se} to add the S-S direct 'dithionic' family and the pentavalent (=O,
# degree-4) As / Sb '-oric' families, so a functional-class derivative of those
# di-acids (diarsoric tetrachloride, distiboric tetraamide, a partial-amido
# dithionic acid,...) builds through the SAME fail-closed, OPSIN-RT-gated
# producer as diphosphoric tetrachloride. Additive: the free parent acids are
# tabled in _INORGANIC_OXOACIDS (matched at name_inorganic_acid's first lookup),
# so they never reach this producer; and the producer runs LAST, only after every
# other acid path abstains, so it can add a name but never regress a passing row.
#
# ONLY the =O-bearing degree-4 families fit the current fixed-oxo/degree-4
# _find_polyacid_backbone shape. The trivalent '-orous'/'-onous' (degree-3, no =O)
# di-acids, the boron 'diboric' (no =O), and 3-centre DERIVATIVE chains remain
# out of the producer (their free acids are B.1 table rows; measured against the
# 3,243 bb_conformance oracle NO gold derivative row needs them — the ch67.2
# derivative gold rows all need isocyanate / trivalent-P-backbone / imido-=N /
# N-locant features that are separately out of scope). They degrade to abstain,
# never to a wrong name.
_POLYACID_PARENT = {
    ("P", "bridge"): "diphosphoric acid",
    ("P", "direct"): "hypodiphosphoric acid",
    ("S", "bridge"): "disulfuric acid",
    ("S", "direct"): "dithionic acid",          # HO-SO2-SO2-OH; 'hypodisulfuric')
    ("Se", "bridge"): "diselenic acid",
    ("As", "bridge"): "diarsoric acid",         # (HO)2As(O)-O-As(O)(OH)2
    ("As", "direct"): "hypodiarsoric acid",     # (HO)2(O)As-As(O)(OH)2
    ("Sb", "bridge"): "distiboric acid",        # (HO)2Sb(O)-O-Sb(O)(OH)2
    ("Sb", "direct"): "hypodistiboric acid",    # (HO)2(O)Sb-Sb(O)(OH)2
}
# =O count that marks a pentavalent (degree-4) acid centre. As/Sb '-oric' centres
# carry exactly one =O, like the -ic phosphorus centre.
_POLYACID_OXO = {"P": 1, "S": 2, "Se": 2, "As": 1, "Sb": 1}
_BRIDGE_THIO_PREFIX = {"S": "thio", "Se": "seleno", "Te": "telluro"}

# Irregular retained oxoanion words — NOT a mechanical 'ic acid'
# -> 'ate' transform (phosphoric->phosphate, sulfuric->sulfate).
_POLYACID_ANION_WORD = {
    "diphosphoric acid": "diphosphate",
    "hypodiphosphoric acid": "hypodiphosphate",
    "disulfuric acid": "disulfate",
    "diselenic acid": "diselenate",
}


def _find_polyacid_backbone(mol):
    """Detect a neutral di-nuclear noncarbon-oxoacid backbone. Return
    ``(parent_name, [c1, c2], bridge_idx_or_None, bridge_elem_or_None)`` or None.

    Acid centres = exactly two same-element P/S/Se atoms each carrying the required
    ``=O`` count. They are either directly bonded (``hypo`` parent) or joined by a
    single bridging atom (O = parent; S/Se/Te = chalcogen-replaced bridge)."""
    centres = []
    for a in atoms_of(mol):
        sym = a.GetSymbol()
        need = _POLYACID_OXO.get(sym)
        if need is None:
            continue
        if a.GetFormalCharge() != 0 or a.GetDegree() != 4:
            continue
        oxo = sum(1 for b in a.GetBonds()
                  if b.GetBondType() == Chem.BondType.DOUBLE
                  and b.GetOtherAtom(a).GetSymbol() == "O")
        if oxo == need:
            centres.append(a)
    if len(centres) != 2:
        return None
    c1, c2 = centres
    if c1.GetSymbol() != c2.GetSymbol():
        return None
    elem = c1.GetSymbol()
    # Direct centre-centre bond -> 'hypo' parent (P only).
    if mol.GetBondBetweenAtoms(c1.GetIdx(), c2.GetIdx()) is not None:
        parent = _POLYACID_PARENT.get((elem, "direct"))
        return (parent, [c1.GetIdx(), c2.GetIdx()], None, None) if parent else None
    # Single bridging atom bonded to both centres.
    n1 = {nb.GetIdx() for nb in c1.GetNeighbors()}
    n2 = {nb.GetIdx() for nb in c2.GetNeighbors()}
    shared = n1 & n2
    if len(shared) != 1:
        return None
    bridge = mol.GetAtomWithIdx(next(iter(shared)))
    if bridge.GetFormalCharge() != 0 or bridge.GetDegree() != 2:
        return None
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in bridge.GetBonds()):
        return None
    belem = bridge.GetSymbol()
    if belem != "O" and belem not in _BRIDGE_THIO_PREFIX:
        return None
    parent = _POLYACID_PARENT.get((elem, "bridge"))
    if parent is None:
        return None
    return (parent, [c1.GetIdx(), c2.GetIdx()], bridge.GetIdx(), belem)


def _classify_acid_position(mol, nb, centre_idx):
    """Classify a single-bonded acid-position neighbour of a polyacid centre.
    Returns ('oh', None) | ('halido', combining) | ('amido', n_prefix_str) |
    ('acyloxy', (acyl_c_idx, o_idx)) | ('cyanatido', None), or None (fail-closed)."""
    sym = nb.GetSymbol()
    if sym == "O":
        if nb.GetFormalCharge() != 0:
            return None
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != centre_idx]
        if nb.GetTotalNumHs() == 1 and not others:
            return ("oh", None)
        if _is_cyanatido_oxygen(mol, nb, centre_idx):
            return ("cyanatido", None)
        if nb.GetDegree() == 2 and len(others) == 1 and others[0].GetSymbol() == "C":
            c = others[0]
            if any(bb.GetBondType() == Chem.BondType.DOUBLE
                   and bb.GetOtherAtom(c).GetSymbol() == "O" for bb in c.GetBonds()):
                return ("acyloxy", (c.GetIdx(), nb.GetIdx()))
        return None
    if sym in _HALIDO_INFIX:
        return ("halido", _HALIDO_INFIX[sym])
    if sym == "N":
        if nb.GetFormalCharge() != 0:
            return None
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in nb.GetBonds()):
            return None
        seg = _amido_n_prefix(mol, nb.GetIdx(), centre_idx)
        return ("amido", seg) if seg is not None else None
    return None


def name_polyacid_derivative(mol) -> Optional[str]:
    """/.2.3/.2.4/.2.5.3: PIN for a di-nuclear noncarbon-oxoacid
    derivative (Engine B). Handles, off a tightly-detected backbone:

      * uniform full replacement -> functional class /.2.4):
        ``diphosphoric tetrachloride`` / ``diphosphoric tetraamide``
      * bridge chalcogen replacement -> prefix at the bridge locant:
        ``2-thiodisulfuric acid``
      * partial replacement -> class prefix on the acid with centre locants
        : ``1,3-diamidodiphosphoric acid``
      * full acyl esterification -> anhydride:
        ``tetraacetic hypodiphosphoric tetraanhydride``

    Fail-closed (returns ``None``) off the tabled backbone set or for any mixed /
    unrecognised acid-position group. Pure: no mol mutation. The free parent acids
    (diphosphoric / disulfuric) are the tabled table rows and never reach here.
    """
    from collections import Counter

    from .functional_replacement import _MULT

    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    bb = _find_polyacid_backbone(mol)
    if bb is None:
        return None
    parent, centres, bridge_idx, bridge_elem = bb
    parent_word = parent[:-len(" acid")]             # 'diphosphoric'

    # Collect classified acid positions per centre (position 1 and 3 for a bridged
    # backbone; 1 and 2 for a direct P-P). Exclude the bridge / other centre / =O.
    exclude = set(centres)
    if bridge_idx is not None:
        exclude.add(bridge_idx)
    positions = {}                                   # centre_idx -> list of (kind, data)
    for c_idx in centres:
        c = mol.GetAtomWithIdx(c_idx)
        plist = []
        for b in c.GetBonds():
            nb = b.GetOtherAtom(c)
            if nb.GetIdx() in exclude:
                continue
            if b.GetBondType() == Chem.BondType.DOUBLE and nb.GetSymbol() == "O":
                continue                             # the centre's =O
            if b.GetBondType() != Chem.BondType.SINGLE:
                return None
            kind = _classify_acid_position(mol, nb, c_idx)
            if kind is None:
                return None
            plist.append(kind)
        positions[c_idx] = plist

    all_kinds = [k for pl in positions.values() for (k, _) in pl]
    if not all_kinds:
        return None
    n_total = len(all_kinds)
    oh_n = all_kinds.count("oh")
    kinds_set = set(all_kinds)

    # Centre locants: bridged -> {1,3}; direct -> {1,2}. (Symmetric backbones make
    # the two orientations equivalent; targets are symmetric.)
    locs = ([1, 3] if bridge_idx is not None else [1, 2])
    centre_loc = {centres[0]: locs[0], centres[1]: locs[1]}
    bridge_loc = 2 if bridge_idx is not None else None

    # --- Case D: full acyl esterification -> anhydride ---
    if kinds_set == {"acyloxy"} and bridge_elem in (None, "O"):
        from .lipids import _acid_fragment_name
        acids = []
        for pl in positions.values():
            for (_, site) in pl:
                acid = _acid_fragment_name(mol, site[0], site[1])
                if not acid or not acid.endswith(" acid"):
                    return None
                acids.append(acid[:-len(" acid")])   # 'acetic acid' -> 'acetic'
        acid_counts = Counter(acids)
        if len(acid_counts) != 1:
            return None                              # mixed acyls -> out of scope
        aname, acnt = next(iter(acid_counts.items()))
        acid_cited = f"{_MULT.get(acnt, '')}{aname}"
        anh_mult = _MULT.get(n_total, "")
        return f"{acid_cited} {parent_word} {anh_mult}anhydride"

    # --- Case A: uniform full replacement -> functional class /.2.4) ---
    if oh_n == 0 and bridge_elem in (None, "O"):
        if kinds_set == {"halido"}:
            hal_words = {"chlorido": "chloride", "fluorido": "fluoride",
                         "bromido": "bromide", "iodido": "iodide"}
            hals = [hal_words[d] for pl in positions.values() for (_, d) in pl]
            hc = Counter(hals)
            if len(hc) != 1:
                return None                          # mixed halides -> defer
            w, cnt = next(iter(hc.items()))
            return f"{parent_word} {_MULT.get(cnt, '')}{w}"
        if kinds_set == {"amido"}:
            # all-bare-amide functional class ('diphosphoric tetraamide'); any
            # N-substituent would need N-locants (out of scope) -> the amido segs
            # must all be empty.
            if any(seg for pl in positions.values() for (_, seg) in pl):
                return None
            return f"{parent_word} {_MULT.get(n_total, '')}amide"
        return None

    # --- Case C / B: >=1 -OH left (class 'acid') -> prefixes on the acid ---
    if oh_n >= 1:
        prefixes = []                                # (sort_key, cited_string)
        # bridge chalcogen replacement -> '{2}-thio' etc.
        if bridge_elem not in (None, "O"):
            pre = _BRIDGE_THIO_PREFIX[bridge_elem]
            prefixes.append((pre, f"{bridge_loc}-{pre}"))
        # class-group replacements on centres, with centre locants
        class_locs = {}                              # combining/base -> [locants]
        class_segs = {}                              # 'amido' -> list of n_prefix segs
        for c_idx, pl in positions.items():
            for (kind, data) in pl:
                if kind == "oh":
                    continue
                if kind == "halido":
                    key = {"chlorido": "chloro", "fluorido": "fluoro",
                           "bromido": "bromo", "iodido": "iodo"}[data]
                elif kind == "amido":
                    key = "amido"
                    class_segs.setdefault("amido", []).append(data)
                elif kind == "cyanatido":
                    key = "cyanato"
                else:
                    return None                      # acyloxy alongside -OH -> defer
                class_locs.setdefault(key, []).append(centre_loc[c_idx])
        for key, ls in class_locs.items():
            ls = sorted(ls)
            # N-substituents on a partial amido would need N-locants -> out of scope
            if key == "amido" and any(class_segs.get("amido", [])):
                return None
            mult = _MULT.get(len(ls), "")
            loc_str = ",".join(str(x) for x in ls)
            prefixes.append((key, f"{loc_str}-{mult}{key}"))
        if not prefixes:
            return None                              # a plain parent acid -> tabled
        prefixes.sort(key=lambda t: t[0])            # alphabetical by prefix stem
        return "".join(p[1] for p in prefixes) + parent


def name_polyacid_anion(frag_mol) -> Optional[str]:
    """: anion word for a (partially or fully) deprotonated
    di-/polynuclear noncarbon-oxoacid whose NEUTRAL parent is a tabled polyacid.
    Re-protonates every terminal ``-O(-)`` to ``-OH``, and iff the result is a
    recognised di-nuclear polyacid backbone with a tabled ``...ic acid`` parent,
    returns the anion word (``'ic acid'`` -> ``'ate'``): ``diphosphate`` /
    ``disulfate`` / ``hypodiphosphate``.

    Fail-closed (returns ``None``) for a non-anion, a mononuclear phosphate/sulfate
    (owned by the ``INORGANIC_ANIONS`` retained table), or any backbone outside the
    tabled polyacid set. Pure: operates on a COPY, no input mutation.
    """
    if frag_mol is None or Chem.GetFormalCharge(frag_mol) >= 0:
        return None
    rw = Chem.RWMol(frag_mol)
    for a in rw.GetAtoms():
        if (a.GetSymbol() == "O" and a.GetFormalCharge() == -1
                and a.GetDegree() == 1):
            a.SetFormalCharge(0)
            a.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    if Chem.GetFormalCharge(rw) != 0:
        return None
    # Must be a genuine di-nuclear polyacid backbone (so mononuclear phosphate /
    # sulfate — handled by INORGANIC_ANIONS — never reaches here).
    if _find_polyacid_backbone(rw) is None:
        return None
    parent = _INORGANIC_OXOACIDS.get(Chem.MolToSmiles(rw))
    # The oxoanion words are irregular retained forms (phosphoric->phosphate,
    # sulfuric->sulfate), so an explicit parent-name -> anion-word map, NOT a naive
    # 'ic acid'->'ate' string transform.
    return _POLYACID_ANION_WORD.get(parent)


# =============================================================================
# a phase Group B: carbonic / poly-carbonic functional-replacement acids
# /.1.3 mononuclear, -.4 di-/tri-/tetra-carbonic).
#
# A CARBON-backbone analogue of name_polyacid_derivative (which is P/S/Se-centred):
# a linear alternating chain of acyl carbons C=X... C=X joined by O/S/Se/NH or
# peroxy (-O-O-) bridges, the two ends bearing an acidic / functional group. Every
# oxygen of the parent poly-carbonic acid may be replaced: a =O by
# =S/=Se/=Te (thio/seleno/telluro), =NH (imido) or =N-NH2 (hydrazono); a bridging
# -O- by -S-/-Se- (thio/seleno), -NH- (imido) or -O-O- (peroxy); an acidic -OH by
# -SH/-SeH/-TeH (thio/seleno/telluro), -OOH (peroxy), -NH2 (amido,,
# -NHNH2 (hydrazido) or a halide. Fully fail-closed by TOTAL heavy-
# atom accounting: any atom the template cannot place -> None (degrade to the
# systematic name; never a wrong constitution). Every emitted name has been
# confirmed to round-trip through OPSIN 2.9.0 to the input structure.
#
# 0-WRONG TAUTOMER GUARD /.3): the nonspecific 'n-thio...' spelling
# is interpreted by OPSIN as the acyl-chalcogen tautomer (=S with -OH retained). So
# a chalcogen on an acidic -XH position is emitted ONLY when the SAME chalcogen also
# sits on that carbon's acyl (both determined, e.g. tetrathio); a determined
# '=O + -SH' carbon (methanethioic S-acid territory, FAILS CLOSED
# rather than emit an ambiguous name that would round-trip to the other tautomer.
# =============================================================================

# Poly-carbonic detachable-prefix words per replacement -.4); NOTE
# these differ from the mononuclear STEM infixes (poly 'peroxy' vs mono 'peroxo').
_CARB_CHALC_PREFIX = {"S": "thio", "Se": "seleno", "Te": "telluro"}
_CARB_HALO_PREFIX = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
# Terminal-only prefixes (cannot sit at a bridge): a lone one on a symmetric parent
# omits its locant; 'chlorodicarbonic acid', not '1-chloro...').
_CARB_TERMINAL_ONLY = set(_CARB_HALO_PREFIX.values()) | {"amido", "hydrazido"}
_CARB_MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa",
              7: "hepta", 8: "octa", 9: "nona", 10: "deca"}


def _carb_acyl(mol, c):
    """The (element, kind) of an acyl carbon's single double-bonded partner, or
    None if the carbon is not a valid acyl carbon. kind in
    {'O','S','Se','Te','imido','hydrazono'}; also returns the consumed heavy-atom
    indices for that acyl group."""
    dbl = [b for b in c.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    if len(dbl) != 1:
        return None
    x = dbl[0].GetOtherAtom(c)
    if x.GetFormalCharge() != 0:
        return None
    sym = x.GetSymbol()
    if sym in ("O", "S", "Se", "Te"):
        if x.GetDegree() != 1 or x.GetTotalNumHs() != 0:
            return None
        return (sym, {x.GetIdx()})
    if sym == "N":
        others = [nb for nb in x.GetNeighbors() if nb.GetIdx() != c.GetIdx()]
        if not others:                                   # =NH -> imido
            return ("imido", {x.GetIdx()}) if x.GetTotalNumHs() == 1 else None
        if len(others) == 1 and others[0].GetSymbol() == "N":   # =N-NH2 -> hydrazono
            n2 = others[0]
            if (mol.GetBondBetweenAtoms(x.GetIdx(), n2.GetIdx()).GetBondType()
                    == Chem.BondType.SINGLE and n2.GetDegree() == 1
                    and n2.GetTotalNumHs() == 2 and n2.GetFormalCharge() == 0
                    and x.GetTotalNumHs() == 0):
                return ("hydrazono", {x.GetIdx(), n2.GetIdx()})
        return None
    return None


def _carb_terminal(mol, c, nb):
    """Classify a single-bonded non-bridge neighbour ``nb`` of acyl carbon ``c`` as
    an acidic / functional terminal group. Returns (kind, consumed_heavy_idxs) with
    kind in {'oh','thio','seleno','telluro','peroxy','amido','hydrazido',
    'fluoro'/'chloro'/'bromo'/'iodo'} or None (fail-closed)."""
    sym = nb.GetSymbol()
    if nb.GetFormalCharge() != 0:
        return None
    if sym in _CARB_HALO_PREFIX:                         # -X halide
        if nb.GetDegree() == 1:
            return (_CARB_HALO_PREFIX[sym], {nb.GetIdx()})
        return None
    if sym in ("O", "S", "Se", "Te"):
        if nb.GetDegree() == 1 and nb.GetTotalNumHs() == 1:   # -OH/-SH/-SeH/-TeH
            return (({"O": "oh", "S": "thio", "Se": "seleno",
                      "Te": "telluro"}[sym]), {nb.GetIdx()})
        if sym == "O" and nb.GetDegree() == 2 and nb.GetTotalNumHs() == 0:
            far = [x for x in nb.GetNeighbors() if x.GetIdx() != c.GetIdx()]
            if (len(far) == 1 and far[0].GetSymbol() == "O"
                    and far[0].GetDegree() == 1 and far[0].GetTotalNumHs() == 1
                    and far[0].GetFormalCharge() == 0):       # -O-OH peroxy
                return ("peroxy", {nb.GetIdx(), far[0].GetIdx()})
        return None
    if sym == "N":
        if nb.GetDegree() == 1 and nb.GetTotalNumHs() == 2:   # -NH2 amido
            return ("amido", {nb.GetIdx()})
        if nb.GetDegree() == 2 and nb.GetTotalNumHs() == 1:   # -NH-NH2 hydrazido
            far = [x for x in nb.GetNeighbors() if x.GetIdx() != c.GetIdx()]
            if (len(far) == 1 and far[0].GetSymbol() == "N"
                    and far[0].GetDegree() == 1 and far[0].GetTotalNumHs() == 2
                    and far[0].GetFormalCharge() == 0):
                return ("hydrazido", {nb.GetIdx(), far[0].GetIdx()})
        return None
    return None


def _carb_bridge(mol, c, nb, acid_cs):
    """If single-bonded neighbour ``nb`` of acyl carbon ``c`` starts a bridge to
    ANOTHER acyl carbon, return (other_c_idx, kind, consumed_heavy_idxs); kind in
    {'O','S','Se','imido','peroxy'}. Else None."""
    sym = nb.GetSymbol()
    if nb.GetFormalCharge() != 0:
        return None
    if sym in ("O", "S", "Se") and nb.GetDegree() == 2 and nb.GetTotalNumHs() == 0:
        far = [x for x in nb.GetNeighbors() if x.GetIdx() != c.GetIdx()]
        if len(far) != 1:
            return None
        w = far[0]
        if w.GetIdx() in acid_cs:                        # single-atom O/S/Se bridge
            return (w.GetIdx(), sym, {nb.GetIdx()})
        if (sym == "O" and w.GetSymbol() == "O" and w.GetDegree() == 2
                and w.GetTotalNumHs() == 0 and w.GetFormalCharge() == 0):
            v = [x for x in w.GetNeighbors() if x.GetIdx() != nb.GetIdx()]
            if len(v) == 1 and v[0].GetIdx() in acid_cs:  # -O-O- peroxy bridge
                return (v[0].GetIdx(), "peroxy", {nb.GetIdx(), w.GetIdx()})
        return None
    if sym == "N" and nb.GetDegree() == 2 and nb.GetTotalNumHs() == 1:
        far = [x for x in nb.GetNeighbors() if x.GetIdx() != c.GetIdx()]
        if len(far) == 1 and far[0].GetIdx() in acid_cs:  # -NH- imido bridge
            return (far[0].GetIdx(), "imido", {nb.GetIdx()})
    return None


def name_carbonic_frn_family(mol):
    """/.1.3 + -.4: PIN for a mononuclear carbonic/carbamic
    functional-replacement acid or a di-/tri-/tetra-carbonic FRN acid, else None.

    Fail-closed by total heavy-atom accounting; pure (no mutation). The plain
    parents (carbonic/dicarbonic acid) and the already-tabled FRN rows are matched
    by the exact table BEFORE this producer, so it only fires on genuine gaps.
    """
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None

    # 1) Acyl carbons: acyclic degree-3 C with exactly one =X (X in O/S/Se/Te/N).
    acid_cs = {}
    for a in atoms_of(mol):
        if a.GetSymbol() != "C" or a.IsInRing() or a.GetFormalCharge() != 0:
            continue
        if a.GetDegree() != 3:
            continue
        acyl = _carb_acyl(mol, a)
        if acyl is None:
            continue
        acid_cs[a.GetIdx()] = acyl
    if not acid_cs:
        return None
    acid_set = set(acid_cs)

    accounted = set()                      # heavy-atom indices placed by the template
    for cidx, (_, acyl_atoms) in acid_cs.items():
        accounted.add(cidx)
        accounted |= acyl_atoms

    # 2) Per carbon: split its two single-bond neighbours into bridges / terminals.
    adj = {c: [] for c in acid_set}        # c -> [(other_c, bridge_kind, atoms)]
    terminals = {c: [] for c in acid_set}  # c -> [(kind, atoms)]
    for cidx in acid_set:
        c = mol.GetAtomWithIdx(cidx)
        for b in c.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                continue
            nb = b.GetOtherAtom(c)
            br = _carb_bridge(mol, c, nb, acid_set)
            if br is not None:
                other, kind, atoms = br
                adj[cidx].append((other, kind, frozenset(atoms)))
                accounted |= atoms
                continue
            term = _carb_terminal(mol, c, nb)
            if term is None:
                return None                # unplaceable neighbour -> fail closed
            kind, atoms = term
            terminals[cidx].append((kind, atoms))
            accounted |= atoms

    # 3) Total heavy-atom accounting -- nothing extra hangs off the skeleton.
    heavy = {a.GetIdx() for a in atoms_of(mol) if a.GetAtomicNum() > 1}
    if accounted != heavy:
        return None

    # 3a) The molecule must RETAIN the acid characteristic group: at least one
    # terminal is an acidic -OH / -SH / -SeH / -TeH / -OOH. If EVERY terminal is a
    # halide / amido / hydrazido, the -OH groups are fully replaced and the compound
    # is an acid HALIDE / AMIDE / HYDRAZIDE functional CLASS (dicarbonyl dichloride,
    # dicarbonic diamide, carbonimidic diamide,...), owned by a different path --
    # fail closed so it is not mis-named as an '-ic acid' principal-group
    # seniority). Without this, Cl-CO-O-CO-Cl and H2N-C(=NH)-NH-CO-NH2 were claimed.
    _ACIDIC = {"oh", "thio", "seleno", "telluro", "peroxy"}
    if not any(k in _ACIDIC for tl in terminals.values() for (k, _) in tl):
        return None

    n = len(acid_set)

    # ---- Mononuclear (n == 1): carbonic / carbamic stem, no locants. ----
    if n == 1:
        cidx = next(iter(acid_set))
        if adj[cidx]:
            return None
        acyl_kind = acid_cs[cidx][0]
        terms = terminals[cidx]
        if len(terms) != 2:               # a carbonic centre has exactly two -X(H)
            return None
        amido = [t for t in terms if t[0] == "amido"]
        if len(amido) == 2:
            return None                   # urea/guanidine diamide -> not an acid here
        # A mononuclear halide / hydrazido / bare-parent is owned elsewhere (tabled
        # carbono-chloridic; carbazic) -> out of this builder's scope.
        stem = "carbam" if len(amido) == 1 else "carbon"
        infixes = {}

        def _bump(key):
            infixes[key] = infixes.get(key, 0) + 1
        # acyl replacement
        if acyl_kind in ("S", "Se", "Te"):
            _bump({"S": "thio", "Se": "seleno", "Te": "telluro"}[acyl_kind])
        elif acyl_kind in ("imido", "hydrazono"):
            _bump(acyl_kind)
        # remaining (non-carbam) terminals
        seen_amido = False
        for kind, _ in terms:
            if kind == "amido" and stem == "carbam" and not seen_amido:
                seen_amido = True
                continue
            if kind == "oh":
                continue
            if kind in ("thio", "seleno", "telluro"):
                _bump(kind)
            elif kind == "peroxy":
                _bump("peroxo")           # mononuclear STEM infix is 'peroxo'
            else:
                return None               # amido#2 / halide / hydrazido -> out of scope
        if not infixes:
            return None                   # plain carbonic/carbamic acid -> tabled
        from .functional_replacement import build_carbonic_mono_frn
        return build_carbonic_mono_frn(stem, infixes)

    # ---- Poly (n >= 2): a linear alternating chain, cited with locants. ----
    if n > 4:
        return None                       # di/tri/tetra-carbonic only (CLOSED set)
    # De-dupe undirected bridges; require exactly n-1 and a simple path.
    bset = set()
    for c, lst in adj.items():
        for other, kind, atoms in lst:
            bset.add((frozenset((c, other)), kind, atoms))
    if len(bset) != n - 1:
        return None
    deg = {c: len(adj[c]) for c in acid_set}
    if sorted(deg.values()) != [1, 1] + [2] * (n - 2):
        return None
    ends = [c for c in acid_set if deg[c] == 1]
    if len(ends) != 2:
        return None
    # Walk the path from one end to build the carbon order + bridge-kind list.
    order = [ends[0]]
    bridges = []                          # bridge kind between order[i], order[i+1]
    prev = None
    cur = ends[0]
    while len(order) < n:
        nxts = [(o, k) for (o, k, _) in adj[cur] if o != prev]
        if len(nxts) != 1:
            return None
        nxt, kind = nxts[0]
        bridges.append(kind)
        order.append(nxt)
        prev, cur = cur, nxt

    def _positions(carbon_order, bridge_kinds):
        """Replacements as {infix: [locants...]} for one numbering direction, or
        None if the 0-wrong tautomer guard trips."""
        reps = {}

        def _add(key, loc):
            reps.setdefault(key, []).append(loc)
        for i, cidx in enumerate(carbon_order):
            pc = 2 * i + 1
            acyl_kind = acid_cs[cidx][0]
            is_terminal = deg[cidx] == 1
            tkinds = [k for (k, _) in terminals[cidx]]
            if acyl_kind in ("S", "Se", "Te"):
                _add(_CARB_CHALC_PREFIX[acyl_kind], pc)
            elif acyl_kind in ("imido", "hydrazono"):
                _add("imido" if acyl_kind == "imido" else "hydrazono", pc)
            for kind in tkinds:
                if kind == "oh":
                    continue
                if kind in ("thio", "seleno", "telluro"):
                    # tautomer guard: a chalcogen on the acidic -XH position is only
                    # unambiguous when the acyl is the SAME chalcogen (both fixed).
                    same = {"thio": "S", "seleno": "Se", "telluro": "Te"}[kind]
                    if acyl_kind != same:
                        return None
                    _add(kind, pc)
                elif kind in ("peroxy", "amido", "hydrazido") or kind in _CARB_HALO_PREFIX.values():
                    _add(kind, pc)
                else:
                    return None
        for j, kind in enumerate(bridge_kinds):
            pb = 2 * j + 2
            if kind == "O":
                continue
            if kind in ("S", "Se"):
                _add(_CARB_CHALC_PREFIX[kind], pb)
            elif kind == "imido":
                _add("imido", pb)
            elif kind == "peroxy":
                _add("peroxy", pb)
            else:
                return None
        return reps

    fwd = _positions(order, bridges)
    rev = _positions(order[::-1], bridges[::-1])
    if fwd is None and rev is None:
        return None

    def _key(reps):
        if reps is None:
            return None
        allpos = sorted(p for ps in reps.values() for p in ps)
        second = tuple((k, sorted(reps[k])) for k in sorted(reps))
        return (allpos, second)
    cands = [r for r in (fwd, rev) if r is not None]
    reps = min(cands, key=_key)

    total_o = 2 * n + 1                    # replaceable O positions of the parent
    n_reps = sum(len(v) for v in reps.values())

    # Assemble the cited-prefix string (alphabetical; /.
    frags = []
    single_infix = len(reps) == 1
    for infix in sorted(reps):
        locs = sorted(reps[infix])
        cnt = len(locs)
        mult = _CARB_MULT.get(cnt)
        if mult is None:
            return None
        omit = (
            (single_infix and cnt == total_o)                  # all positions same
            or (n_reps == 1 and infix in _CARB_TERMINAL_ONLY)  # lone terminal-only
        )
        if omit:
            frags.append(f"{mult}{infix}")
        else:
            frags.append(f"{','.join(str(x) for x in locs)}-{mult}{infix}")
    prefix = "-".join(frags)
    return f"{prefix}{_CARB_MULT[n]}carbonic acid"


# --- Substituted / multiplicative hydrazinecarboxamide (semicarbazide) family ---
# (BB 32675) 'carboxamide' + (systematic name is the
# PIN) + (multiplicative). The exact base 'NNC(N)=O' -> 'hydrazinecarboxamide'
# is tabled above, but every SUBSTITUTED form and the symmetric bis-form (two units
# joined through their amide nitrogens by an alkanediyl) were mis-perceived as a
# 'hydrazide' and garbled. This general namer perceives the
# R2N-N(R)-C(=O)-N(R)R skeleton, assigns the N / 1 / 2 locants (carboxamide N = 'N',
# hydrazine N1 = the C-bonded hydrazine N, hydrazine N2 = the terminal one), and
# names either the substituted monomer or the multiplicative dimer. Fails
# closed on any senior group, unaccounted atom, or unnameable substituent.
_HZC_MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}


def _hzc_branch_atoms(mol, start, banned):
    """Atoms of the branch reachable from ``start`` without crossing ``banned``."""
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j not in banned and j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def _hzc_find_centers(mol):
    """Each hydrazinecarboxamide center as ``{C,O,N,N1,N2}``. Fail-closed shape:
    an acyclic sp2 C with exactly one =O, two single-bonded N neighbours, no other
    heavy neighbour; one N carries a further N-N (hydrazine N1 -> its terminal
    partner is N2), the other is the amide N."""
    centers = []
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'C' or atom.IsInRing():
            continue
        ci = atom.GetIdx()
        o_dbl, n_single, other = [], [], []
        for n in atom.GetNeighbors():
            j = n.GetIdx()
            bt = mol.GetBondBetweenAtoms(ci, j).GetBondType()
            if n.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(
                    ci, j).GetBondTypeAsDouble() == 2.0:
                o_dbl.append(j)
            elif n.GetSymbol() == 'N' and bt == Chem.BondType.SINGLE:
                n_single.append(j)
            elif n.GetAtomicNum() > 1:
                other.append(j)
        if len(o_dbl) != 1 or len(n_single) != 2 or other:
            continue
        hy = []
        for ni in n_single:
            n_neigh = [x.GetIdx() for x in mol.GetAtomWithIdx(ni).GetNeighbors()
                       if x.GetSymbol() == 'N' and x.GetIdx() != ci]
            hy.append((ni, n_neigh))
        with_nn = [(ni, nn) for ni, nn in hy if nn]
        without_nn = [ni for ni, nn in hy if not nn]
        if len(with_nn) != 1 or len(without_nn) != 1:
            continue
        n1, n1_nn = with_nn[0]
        if len(n1_nn) != 1:
            continue
        n2 = n1_nn[0]
        a_n2 = mol.GetAtomWithIdx(n2)
        if a_n2.IsInRing() or any(
                x.GetSymbol() == 'N' and x.GetIdx() != n1
                for x in a_n2.GetNeighbors()):
            continue
        # The hydrazine N2 is an amine nitrogen; its ONLY permitted multiple bond
        # is a C=N (semicarbazone -> '2-ylidene'). A double bond to O (nitroso ->
        # N-nitrosourea) or N (azo) is a DIFFERENT parent -- fail closed so those
        # keep their own PIN. N1 must likewise be a plain amine (all single).
        _bad_n2 = any(
            b.GetBondTypeAsDouble() >= 2.0
            and mol.GetAtomWithIdx(
                b.GetOtherAtomIdx(n2)).GetSymbol() != 'C'
            for b in a_n2.GetBonds())
        _bad_n1 = any(
            b.GetBondTypeAsDouble() >= 2.0 for b in mol.GetAtomWithIdx(n1).GetBonds())
        if _bad_n2 or _bad_n1:
            continue
        centers.append({'C': ci, 'O': o_dbl[0],
                        'N': without_nn[0], 'N1': n1, 'N2': n2})
    return centers


def _hzc_unit_substituents(mol, center, extra_banned):
    """``[(locant, token), …]`` for the N/N1/N2 substituents of one unit, or None
    if any is unnameable. ``extra_banned`` excludes the multiplicative bridge."""
    from ..assembly.substituent_enumerator import name_substituent
    from ..errors import is_refusal_sentinel
    skeleton = {center['C'], center['O'], center['N'],
                center['N1'], center['N2']}
    out = []
    for n_atom, locant in ((center['N'], 'N'), (center['N1'], '1'),
                           (center['N2'], '2')):
        for nb in mol.GetAtomWithIdx(n_atom).GetNeighbors():
            j = nb.GetIdx()
            if j in skeleton or j in extra_banned or nb.GetAtomicNum() <= 1:
                continue
            frag = _hzc_branch_atoms(mol, j, {n_atom} | extra_banned)
            tok = name_substituent(mol, sorted(frag), j)
            if not tok or is_refusal_sentinel(tok) or ' ' in tok:
                return None
            out.append((locant, tok, frozenset(frag)))
    return out


def _hzc_assemble_prefix(subs):
    """Alpha-ordered multiplied substituent prefix (P-14.5.2). ``subs`` is a list
    of ``(locant, token, …)``; identical tokens across N/1/2 combine."""
    from collections import defaultdict

    from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound
    groups = defaultdict(list)
    for entry in subs:
        groups[entry[1]].append(entry[0])

    def _loc_key(l):
        return (0, 0) if l == 'N' else (1, int(l))
    items = []
    for tok, locs in groups.items():
        locs_sorted = sorted(locs, key=_loc_key)
        mult = _HZC_MULT.get(len(locs), '')
        #: a compound/complex substituent (internal locant, two prefixes,
        # inner mark) takes enclosing marks -- '2-(propan-2-ylidene)', not
        # '2-propan-2-ylidene'.
        rendered = enclose_if_compound(tok)
        items.append((alpha_sort_key(tok),
                      f"{','.join(locs_sorted)}-{mult}{rendered}"))
    items.sort(key=lambda x: x[0])
    return '-'.join(r for _, r in items)


def _hzc_monomer_body(mol, center, extra_banned):
    """The (substituent-prefix, is_bare) rendering of one hydrazinecarboxamide
    unit, or None. ``is_bare`` -> use 'hydrazinecarboxamide' (no locant)."""
    subs = _hzc_unit_substituents(mol, center, extra_banned)
    if subs is None:
        return None
    if not subs:
        return ("hydrazinecarboxamide", frozenset())
    prefix = _hzc_assemble_prefix(subs)
    consumed = frozenset().union(*(e[2] for e in subs))
    return (f"{prefix}hydrazine-1-carboxamide", consumed)


def name_hydrazinecarboxamide(mol) -> Optional[str]:
    """PIN for a substituted / multiplicative hydrazinecarboxamide, or None."""
    from ..data.chain_names import get_chain_prefix
    if mol is None:
        return None
    centers = _hzc_find_centers(mol)
    if not centers:
        return None
    total_heavy = {a.GetIdx() for a in atoms_of(mol) if a.GetAtomicNum() > 1}

    # Seniority guard: the carboxamide is only the PIN parent when nothing
    # senior is present. A carbon carbonyl / nitrile / an S or P oxoacid OUTSIDE
    # the center skeleton(s) would outrank the carboxamide, so fail closed and let
    # the general seniority machinery own the molecule (avoids a wrong parent).
    _center_c = {c['C'] for c in centers}
    for bond in bonds_of(mol):
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        dbl = bond.GetBondTypeAsDouble()
        for x, y in ((a, b), (b, a)):
            if (x.GetSymbol() == 'C' and x.GetIdx() not in _center_c
                    and ((y.GetSymbol() in ('O', 'S') and dbl == 2.0)
                         or (y.GetSymbol() == 'N' and dbl == 3.0))):
                return None
    if any(at.GetSymbol() in ('S', 'P') for at in atoms_of(mol)):
        return None

    if len(centers) == 1:
        c = centers[0]
        body = _hzc_monomer_body(mol, c, set())
        if body is None:
            return None
        name, consumed = body
        accounted = {c['C'], c['O'], c['N'], c['N1'], c['N2']} | set(consumed)
        if accounted != total_heavy:
            return None  # unaccounted atom / senior group -> fail closed
        return name

    if len(centers) == 2:
        c1, c2 = centers
        # The bridge is a single unbranched all-carbon acyclic chain joining the
        # two amide nitrogens multiplicative linker).
        start = next((x.GetIdx() for x in mol.GetAtomWithIdx(c1['N']).GetNeighbors()
                      if x.GetSymbol() == 'C' and x.GetIdx() != c1['C']), None)
        if start is None:
            return None
        chain = []
        prev, cur = c1['N'], start
        seen = {c1['N']}
        while True:
            a = mol.GetAtomWithIdx(cur)
            if a.GetSymbol() != 'C' or a.IsInRing():
                return None
            chain.append(cur)
            seen.add(cur)
            nxt = [x.GetIdx() for x in a.GetNeighbors()
                   if x.GetIdx() != prev and x.GetAtomicNum() > 1]
            if len(nxt) != 1:
                return None  # branch or dead-end -> not a clean diyl bridge
            nn = nxt[0]
            if nn == c2['N']:
                break
            if nn in seen:
                return None
            prev, cur = cur, nn
        bridge = set(chain)
        # each amide N attaches to the bridge only (no other substituent), and the
        # two monomer bodies (N1/N2 substituents) must be identical -> 'bis'.
        body1 = _hzc_monomer_body(mol, c1, bridge)
        body2 = _hzc_monomer_body(mol, c2, bridge)
        if body1 is None or body2 is None or body1[0] != body2[0]:
            return None
        name1, cons1 = body1
        _, cons2 = body2
        accounted = (bridge
                     | {c1['C'], c1['O'], c1['N'], c1['N1'], c1['N2']}
                     | {c2['C'], c2['O'], c2['N'], c2['N1'], c2['N2']}
                     | set(cons1) | set(cons2))
        if accounted != total_heavy:
            return None
        stem = get_chain_prefix(len(chain))
        if not stem:
            return None
        diyl = f"{stem}ane-1,{len(chain)}-diyl"
        return f"N,N'-({diyl})bis({name1})"

    return None


def name_inorganic_acid(mol) -> Optional[str]:
    """Return the PIN for a free inorganic oxoacid or its tabled functional-class
    derivative (acid halide / amide / carbonic-FRN acid), the tetraalkyl silicate
    ester, else ``None``.

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
    # Substituted / multiplicative hydrazinecarboxamide (semicarbazide family):
    # the exact base is tabled above; the substituted and bis forms are
    # named here (fail-closed graph matcher, zero false positives by full atom
    # accounting).
    hzc = name_hydrazinecarboxamide(mol)
    if hzc is not None:
        return hzc
    # a phase /.1.3 + -.4): carbonic / poly-carbonic
    # functional-replacement acids (carbonimidothioic acid, 2-imidodicarbonic acid,
    # 1-amido-2-thiodicarbonic acid,...). Fail-closed by total heavy-atom
    # accounting; the plain parents + tabled FRN rows matched above never reach it.
    carb_frn = name_carbonic_frn_family(mol)
    if carb_frn is not None:
        return carb_frn
    # W3-P10: mononuclear-P oxoacid functional-replacement (Engine A):
    # phosphoramidic / phosphonocyanatidic / phosphonochloridic acids. Fires only on
    # an acid (>=1 -OH) carrying a class replacement group, so plain phosphonic /
    # phosphoric acids (no class group) and the amide/acid-halide functional classes
    # (no -OH) never match here and keep their existing paths.
    frn = name_p_oxoacid_frn(mol)
    if frn is not None:
        return frn
    # a phase: chalcogen-infix FRN, locant-free di+ subset
    # (arsorodithioic / phosphorotrithioic). Fires only on >=2 identical -SH/-SeH/-TeH
    # with a =O; mono / tautomer-locant cases fail closed (a phase's.
    chalco_frn = name_p_oxoacid_chalcogen_frn(mol)
    if chalco_frn is not None:
        return chalco_frn
    # a phase /: mononuclear P/As/Sb oxoacid HALIDE /
    # AMIDE (oh_count == 0). Fires only when every -OH is replaced, so the acid
    # paths above (>=1 -OH) never reach here; fail-closed on the phosphoryl/sulfuryl
    # acyl-word exception so POCl3 stays 'phosphoryl trichloride'.
    hal_amide = name_p_oxoacid_halide_amide(mol)
    if hal_amide is not None:
        return hal_amide
    # a phase: FRN-modified phosphorus-acid ESTER (>=1 -O-R owner
    # AND >=1 class infix): methyl N,N-dimethylphosphoramidochloridite. A
    # non-overlapping addition; a plain phosphate/phosphite ester never reaches here.
    p_frn_ester = name_p_frn_ester(mol)
    if p_frn_ester is not None:
        return p_frn_ester
    # W3-P10: chalcogen-prefix FRN on silicic acid (thiosilicic).
    si_frn = name_silicic_acid_frn(mol)
    if si_frn is not None:
        return si_frn
    # W3-P10: boron-acid amides named on borane (boranetriamine).
    b_amine = name_borane_amine(mol)
    if b_amine is not None:
        return b_amine
    # W3-P10: mixed acyl/phosphoric anhydride named substitutively as
    # an (acyloxy)phosphonic acid (acid senior to anhydride).
    from .phosphorus import name_acyloxy_phosphonic_acid
    axp = name_acyloxy_phosphonic_acid(mol)
    if axp is not None:
        return axp
    # W3-P10.x): di-/polynuclear oxoacid derivatives (Engine B):
    # diphosphoric tetrachloride/tetraamide, 2-thiodisulfuric acid,
    # 1,3-diamidodiphosphoric acid, tetraacetic hypodiphosphoric tetraanhydride.
    poly = name_polyacid_derivative(mol)
    if poly is not None:
        return poly
    # D-FOLLOWON item 10: tetraalkyl silicate ester (Si(OR)4). Routed here (@40)
    # so it intercepts BEFORE ORGANOMETALLIC@50 (which mis-claims Si as a metalloid
    # hub and linearizes the silyl-ester ligands into nonsense).
    return name_silicate_ester(mol)


def name_azinic_derivative(mol) -> Optional[str]:
    """ ylidene derivatives of azinic acid H2N(O)OH (Wave-2
    completion C): CH3-CH=N(O)-OH -> ethylideneazinic acid (BB verbatim;
    aci-nitroethane). Fail-closed: exactly one N+ with a -O(-), an -OH and a
    DOUBLE bond to an unbranched all-carbon chain; net charge 0; nitro
    tautomers (=O on N) and nitronate anions never match. Pure."""
    from rdkit import Chem
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    ns = [a for a in atoms_of(mol)
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
    if any(a.GetFormalCharge() != 0 for a in atoms_of(mol)
           if a.GetIdx() not in (n.GetIdx(), o_minus.GetIdx())):
        return None
    from ..assembly.naming_utils import unbranched_alkylidene_name
    ylidene = unbranched_alkylidene_name(mol, c.GetIdx(), n.GetIdx())
    if ylidene is None:
        return None
    if mol.GetNumHeavyAtoms() != 3 + sum(
            1 for a in atoms_of(mol) if a.GetAtomicNum() == 6):
        return None
    return f"{ylidene}azinic acid"
