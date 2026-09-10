"""Shared functional-replacement (FRN) acid-name engine — P-67.1.2 / P-65.2.1.2.

ONE name-builder for the functional-replacement / functional-class derivatives of
the retained acid *parents* (carbonic, carbamic, phosphoric, sulfuric, …), used by
BOTH the carbon acids (P-65.2 carbonic family) and the non-carbon oxoacids (P-67).
Per the cross-cutting design (§7, FRN-infix owner = a phase) the construction
logic lives here once, not duplicated per family.

This module is PURE string assembly: it does NOT perceive structure. The caller
(``rules.inorganic_acids`` / future P-67 non-carbon adapters) decides *which*
replacement a given structure represents — keyed on the exact canonical SMILES,
fail-closed — and asks this engine to spell the name. Keeping perception out of
the builder is what makes it safely reusable and free of false positives.

Coverage in a phase = the OPSIN-round-trippable plain forms:
  * chalcogen / peroxo infix replacement on a retained acid (``carbonoperoxoic``,
    ``carbonodithioic``, ``carbonotrithioic``)
  * the =O -> =NH imido replacement (``carbonimidic``,
    ``carbamimidic``)
  * multiplicative poly-acids (``dicarbonic``,
    ``tricarbonic``)
  * acyl-halide functional-class words (``phosphoryl
    trichloride``, ``sulfuryl dichloride``)
The italic O/S/Se tautomer-locant word-forms (``carbonothioic S-acid``) are
DEFERRED to a phase — OPSIN rejects the word-form, so they require name-exact
gold rather than round-trip validation (V23 plan §6, a phase).
"""
import re
from typing import Dict, Optional

# Numerical multiplying prefixes (P-14.2). Index = count.
_MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}

# --- a phase: P-67.1.4.1.1 ACYL-PREFIX TABLE (SHARED — a phase consumes it) -
# P-67.1.4.1.1.2/.1.4: an acyl prefix is formed from the acid name by changing
# 'ic acid' -> 'oyl', EXCEPT the retained -oryl / nitroryl forms for the -oric /
# nitric (0-skeletal =O) acids (BB L36037-36046: the rule would give 'phosphoroyl'
# but 'phosphoryl' is retained). Keyed by (element, =E token, skeletal C+H count on
# the central atom): 0 -> the -oryl exception (or -oyl for thio/imido/nitrido), 1 ->
# -onoyl, 2 -> -inoyl. This table is chapter-67-owned; a phase (P-68.3.2.3.2.2)
# looks up (hydroxyarsoryl) / (dimethylphosphinothioyl) / (phosphonimidoyl) here
# rather than re-deriving the acyl names. Provenance per cell: (BB) = verbatim from
# a P-67.1.4.1.1 example; (rule) = the 'ic acid'->'oyl' rule applied to a real acid.
ACYL_PREFIX_TABLE = {
    # =O fundamental acyl groups (P-67.1.4.1.1.2)
    ("P", "oxo", 0): "phosphoryl",       # phosphoric (BB L36044, retained -oryl)
    ("P", "oxo", 1): "phosphonoyl",      # phosphonic (BB L36049)
    ("P", "oxo", 2): "phosphinoyl",      # phosphinic (BB L36050)
    ("As", "oxo", 0): "arsoryl",         # arsoric (BB L36045, retained -oryl)
    ("As", "oxo", 1): "arsonoyl",        # arsonic (rule)
    ("As", "oxo", 2): "arsinoyl",        # arsinic (rule)
    ("Sb", "oxo", 0): "stiboryl",        # stiboric (BB L36046, retained -oryl)
    ("Sb", "oxo", 1): "stibonoyl",       # stibonic (rule)
    ("Sb", "oxo", 2): "stibinoyl",       # stibinic (rule)
    # =S thioacyl groups (P-67.1.4.1.1.4)
    ("P", "thio", 0): "phosphorothioyl", # (BB L36132; 'thiophosphoryl' is the alt)
    ("P", "thio", 1): "phosphonothioyl", # (rule)
    ("P", "thio", 2): "phosphinothioyl", # phosphinothioic -> -oyl (rule; BB L35724 acid)
    ("As", "thio", 0): "arsorothioyl",   # (rule)
    # =NH imidoacyl groups (P-67.1.4.1.1.4)
    ("P", "imido", 0): "phosphorimidoyl",  # (rule)
    ("P", "imido", 1): "phosphonimidoyl",  # (rule; BB L35610 acid phosphinimidic sibling)
    ("P", "imido", 2): "phosphinimidoyl",  # (BB L36150)
    ("As", "imido", 0): "arsorimidoyl",    # (BB L36136)
    # =N (nitrido) acyl group (P-67.1.4.1.1.4)
    ("P", "nitrido", 0): "phosphoronitridoyl",  # (BB L36162)
}


def acyl_prefix_for(element: str, chalcogen: str = "oxo",
                    skeletal: int = 0) -> Optional[str]:
    """P-67.1.4.1.1 acyl PREFIX base name for a mononuclear P/As/Sb acid core, or
    ``None`` if the cell is not tabled (fail-closed). ``chalcogen`` is the =E token
    ('oxo'/'thio'/'imido'/'nitrido'); ``skeletal`` is the C + H count on the central
    atom (0 -> the -oryl exception, 1 -> -onoyl, 2 -> -inoyl). SHARED export: a phase
    (P-68.3.2.3.2.2) must look up acyl names here, never re-spell them.

        acyl_prefix_for("P", "oxo", 0) -> "phosphoryl"
        acyl_prefix_for("As", "oxo", 0) -> "arsoryl"
        acyl_prefix_for("P", "thio", 2) -> "phosphinothioyl"
    """
    return ACYL_PREFIX_TABLE.get((element, chalcogen, skeletal))

# --- W3-P10 (P-67.1.2.3.2 / P-67.1.2.4): class infixes for the mononuclear-P
# oxoacid functional-replacement engine. Each combining form carries a trailing
# linking ``o`` that is elided before a vowel (P-67.1.2.3.5). The chalcogen /
# peroxo infixes (thio/seleno/telluro/peroxo) do NOT belong here — they never
# elide their ``o`` and are spelled by:func:`build_frn_acid_name`. Keyed by the
# combining form so the builder both validates the infix and orders it
# alphabetically. ---
_CLASS_INFIX = {
    # halido (P-67.1.2.3.2 (1); cited in alphabetical order among themselves)
    "bromido", "chlorido", "fluorido", "iodido",
    # pseudohalido (P-67.1.2.3.2 (2); alphabetical)
    "azido", "cyanatido", "cyanido", "isocyanatido", "isocyanido",
    "isothiocyanatido", "thiocyanatido",
    # amido / hydrazido / nitrido (P-67.1.2.3.2 (3)-(5))
    "amido", "hydrazido", "nitrido",
}


def _elide_o_before_vowel(word: str) -> str:
    """P-67.1.2.3.5: elide a linking ``o`` immediately before a vowel. Applied to
    the class-infix acid stems only (chalcogen infixes are never routed here, so
    their protected ``o`` is untouched)."""
    return re.sub(r"o(?=[aeiou])", "", word)


def build_p_frn_acid_name(front_prefix: str, parent_stem: str,
                          infix_counts: Dict[str, int]) -> Optional[str]:
    """Spell a mononuclear noncarbon-oxoacid functional-replacement acid
    (P-67.1.2.4) whose replacements are class infixes (amido / halido /
    pseudohalido), NOT chalcogen infixes.

    ``front_prefix`` — already-assembled detachable-prefix string cited in front
                        (organyl on P: ``"methyl"`` / ``"phenyl"``; or the
                        N-locant amido substituents: ``"N,N-dimethyl"``). ``""``
                        for the bare parent.
    ``parent_stem`` — parent-acid stem WITHOUT its linking vowel: ``"phosphor"``
                        (phosphoric), ``"phosphon"`` (phosphonic), ``"phosphin"``
                        (phosphinic), ``"arsor"``/``"arson"`` etc.
    ``infix_counts`` — ``{combining-form: multiplicity}`` for the class infixes
                        present, e.g. ``{"amido": 1}`` / ``{"chlorido": 1}`` /
                        ``{"cyanatido": 1}``.

    Returns ``None`` (fail-closed) for an unknown infix or an out-of-range count.

    Examples::

        build_p_frn_acid_name("N,N-dimethyl", "phosphor", {"amido": 1})
            -> "N,N-dimethylphosphoramidic acid"
        build_p_frn_acid_name("methyl", "phosphon", {"cyanatido": 1})
            -> "methylphosphonocyanatidic acid"
        build_p_frn_acid_name("phenyl", "phosphon", {"chlorido": 1})
            -> "phenylphosphonochloridic acid"
    """
    stem_word = build_p_frn_acid_stem_word(front_prefix, parent_stem, infix_counts)
    if stem_word is None:
        return None
    return f"{stem_word} acid"


def build_p_frn_acid_stem_word(front_prefix: str, parent_stem: str,
                               infix_counts: Dict[str, int]) -> Optional[str]:
    """The class-infix FRN acid name WITHOUT the trailing ``" acid"`` — i.e. the
    ``"{front}{stem}o{infixes}ic"`` stem word that:func:`build_p_frn_acid_name`
    appends ``" acid"`` to, and that the halide / amide builders append a class
    word to (P-67.1.2.5 / P-67.1.2.6 derive the halide/amide from the same acid
    stem). Fail-closed (``None``) for an unknown infix or out-of-range count.

        build_p_frn_acid_stem_word("N,N-dimethyl", "phosphor", {"amido": 1})
            -> "N,N-dimethylphosphoramidic"
    """
    parts = [parent_stem + "o"]                       # phosphor -> phosphoro
    for infix in sorted(infix_counts):                # alphabetical (P-67.1.2.3.5)
        if infix not in _CLASS_INFIX:
            return None
        cnt = infix_counts[infix]
        if cnt not in _MULT:
            return None
        parts.append(f"{_MULT[cnt]}{infix}")
    stem_word = _elide_o_before_vowel("".join(parts) + "ic")
    return f"{front_prefix}{stem_word}"

# FRN infix combining forms inserted before the ``-ic acid`` ending
# (P-67.1.2.3 / P-65.2.1.2). The chalcogen/peroxo infixes take the linking ``o``
# of the parent stem; the nitrogen infix ``imido`` is contracted to ``imid`` and
# attaches WITHOUT the linking ``o`` (``carbonimidic``, not ``carbonoimidic``).
_CHALCOGEN_INFIX = {"thio", "peroxo", "seleno", "telluro"}
_IMIDO_INFIX = {"imido": "imid", "hydrazono": "hydrazon"}


def build_frn_acid_name(base_stem: str, infix: str, count: int = 1) -> Optional[str]:
    """Spell a single-infix functional-replacement acid (``-ic acid`` parents).

    ``base_stem`` is the parent-acid stem WITHOUT its trailing linking vowel:
    ``"carbon"`` (carbonic), ``"carbam"`` (carbamic), ``"phosphor"`` (phosphoric),
    ``"sulfur"`` (sulfuric). ``infix`` is the FRN infix key; ``count`` its
    multiplicity. Returns ``None`` for an unknown infix (fail-closed).

    Examples::

        build_frn_acid_name("carbon", "peroxo", 1) -> "carbonoperoxoic acid"
        build_frn_acid_name("carbon", "thio", 2) -> "carbonodithioic acid"
        build_frn_acid_name("carbon", "thio", 3) -> "carbonotrithioic acid"
        build_frn_acid_name("carbon", "imido", 1) -> "carbonimidic acid"
        build_frn_acid_name("carbam", "imido", 1) -> "carbamimidic acid"
    """
    if count not in _MULT:
        return None
    if infix in _CHALCOGEN_INFIX:
        # carbon + o + [di|tri] + thio + ic acid
        return f"{base_stem}o{_MULT[count]}{infix}ic acid"
    if infix in _IMIDO_INFIX:
        # carbon + imid + ic acid (linking o elided; multiplier rare, supported)
        return f"{base_stem}{_MULT[count]}{_IMIDO_INFIX[infix]}ic acid"
    return None


# --- a phase (P-65.2.1.2/.1.3): MULTI-infix mononuclear carbonic/carbamic FRN
# acid. Generalises build_frn_acid_name to a co-occurring set of =X (imido /
# hydrazono) and chalcogen/peroxo (thio/seleno/telluro/peroxo) replacements on ONE
# carbonic/carbamic centre, e.g. carbonimidothioic acid (HS-C(=NH)-OH) and
# carbonohydrazonodiselenoic acid (HSe-C(=NNH2)-SeH). The =X infixes contract to
# 'imid'/'hydrazon' ONLY when they are the LAST component before '-ic'; medially
# they keep their 'o' (carbonimido-thio-ic). The stem linking 'o' is dropped before
# a vowel (carbono+imido -> carbonimido). Infix set is CLOSED (P-65.2.1.3). ---
_CARBONIC_NX_INFIX = {"imido": ("imido", "imid"),      # (medial, final-before-ic)
                      "hydrazono": ("hydrazono", "hydrazon")}


def build_carbonic_mono_frn(base_stem: str,
                            infix_counts: Dict[str, int]) -> Optional[str]:
    """Spell a mononuclear carbonic/carbamic functional-replacement acid carrying a
    SET of FRN infixes (P-65.2.1.2/.1.3). ``base_stem`` is ``"carbon"`` or
    ``"carbam"``; ``infix_counts`` maps an infix in
    ``{thio, seleno, telluro, peroxo, imido, hydrazono}`` to its multiplicity.
    Infixes are cited in ALPHABETICAL order (P-65.2.1.3 / P-14.5.2), the =X infix
    contracted only when it is final. Fail-closed (``None``) for an unknown infix,
    an out-of-range count, or an empty set.

        build_carbonic_mono_frn("carbon", {"imido": 1, "thio": 1})
            -> "carbonimidothioic acid"
        build_carbonic_mono_frn("carbam", {"peroxo": 1}) -> "carbamoperoxoic acid"
        build_carbonic_mono_frn("carbon", {"peroxo": 2}) -> "carbonodiperoxoic acid"
        build_carbonic_mono_frn("carbon", {"hydrazono": 1})-> "carbonohydrazonic acid"
        build_carbonic_mono_frn("carbam", {"imido": 1, "seleno": 1})
            -> "carbamimidoselenoic acid"
        build_carbonic_mono_frn("carbon", {"hydrazono": 1, "seleno": 2})
            -> "carbonohydrazonodiselenoic acid"
    """
    if base_stem not in ("carbon", "carbam") or not infix_counts:
        return None
    order = sorted(infix_counts)                       # alphabetical (P-14.5.2)
    tokens = []
    for i, infix in enumerate(order):
        cnt = infix_counts[infix]
        if cnt not in _MULT:
            return None
        mult = _MULT[cnt]
        is_last = (i == len(order) - 1)
        if infix in _CHALCOGEN_INFIX:
            tokens.append(f"{mult}{infix}")            # keeps its own linking 'o'
        elif infix in _CARBONIC_NX_INFIX:
            medial, final = _CARBONIC_NX_INFIX[infix]
            tokens.append(f"{mult}{final if is_last else medial}")
        else:
            return None
    body = "".join(tokens)
    # Stem linking 'o', elided before a vowel (carbono+imido -> carbonimido). The
    # base stems end in a consonant, so only a vowel-initial body (imido) elides.
    linked = base_stem + body if body[0] in "aeiou" else base_stem + "o" + body
    return f"{linked}ic acid"


# --- a phase: acid-HALIDE / -AMIDE functional-class builders (P-67.1.2.5 /
# P-67.1.2.6). Siblings of build_p_frn_acid_name: same acid stem word, but the
# molecule has NO -OH left (oh_count == 0) so it is not class 'acid' — the class
# word is a halide / pseudohalide / amide / hydrazide instead. SHARED with a phase
# (P-65 chalcogen acid halides/amides). Pure string assembly; the caller perceives
# the shape and assembles the acid stem word (via build_p_frn_acid_name minus its
# ' acid', or a bare -ous / -ic / -amidic stem word). ---

# P-67.1.2.6.1(b) seniority order of the halide / pseudohalide class words, most
# senior first; several are cited in this order (P-67.1.5.2 for the acid, this list
# for the class words). Verbatim from the BB list "Br, Cl, F, I, N3, CN, NC, NCO,
# ONC" (BB L35769). 'diphenylphosphinous chloride', 'phenylphosphonous bromide
# chloride' (BB L35712, L35718): different halides are each a class word, cited in
# this order; identical halides are multiplied ('dichloride').
_HALIDE_CLASS_ORDER = (
    "bromide", "chloride", "fluoride", "iodide",          # halides
    "azide", "cyanide", "isocyanide",                     # pseudohalides
    "isocyanate", "cyanate", "isothiocyanate", "thiocyanate",
)
_HALIDE_CLASS_RANK = {w: i for i, w in enumerate(_HALIDE_CLASS_ORDER)}


def build_p_frn_halide_name(acid_stem_word: str,
                            halide_counts: Dict[str, int]) -> Optional[str]:
    """Spell a mononuclear noncarbon-oxoacid HALIDE / pseudohalide (P-67.1.2.5.1).

    ``acid_stem_word`` — the acid name with the trailing ``" acid"`` stripped:
                          ``"phenylphosphonous"`` (BB ``phenylphosphonous
                          dichloride``), ``"phenylphosphonic"``,
                          ``"diphenylphosphinous"``, or an FRN-infixed stem such
                          as ``"N,N-dimethylphosphoramidic"`` (BB
                          ``N,N-dimethylphosphoramidic dichloride``).
    ``halide_counts`` — ``{class-word: multiplicity}`` for the halide /
                          pseudohalide principal groups, e.g. ``{"chloride": 2}``
                          or ``{"bromide": 1, "chloride": 1}``. The class words are
                          cited in:data:`_HALIDE_CLASS_ORDER`; identical ones are
                          multiplied (P-67.1.2.5.1).

    Returns ``None`` (fail-closed) for an unknown class word or out-of-range count.

    Examples::

        build_p_frn_halide_name("phenylphosphonous", {"chloride": 2})
            -> "phenylphosphonous dichloride"
        build_p_frn_halide_name("phenylphosphonous", {"bromide": 1, "chloride": 1})
            -> "phenylphosphonous bromide chloride"
        build_p_frn_halide_name("N,N-dimethylphosphoramidic", {"chloride": 2})
            -> "N,N-dimethylphosphoramidic dichloride"
    """
    if not acid_stem_word or not halide_counts:
        return None
    words = []
    for cls in sorted(halide_counts, key=lambda w: _HALIDE_CLASS_RANK.get(w, 999)):
        if cls not in _HALIDE_CLASS_RANK:
            return None
        cnt = halide_counts[cls]
        if cnt not in _MULT:
            return None
        words.append(f"{_MULT[cnt]}{cls}")
    return f"{acid_stem_word} {' '.join(words)}"


def build_p_frn_amide_name(acid_stem_word: str, kind: str = "amide",
                           count: int = 1) -> Optional[str]:
    """Spell a mononuclear noncarbon-oxoacid AMIDE / hydrazide (P-67.1.2.6.1).

    ``acid_stem_word`` — the acid name minus ``" acid"`` (including any N-/P-
                          substituent prefixes already assembled), e.g.
                          ``"N,N,P,P-tetramethylphosphinic"`` (BB
                          ``N,N,P,P-tetramethylphosphinic amide``) or
                          ``"P-phenylphosphonic"`` for a diamide.
    ``kind`` — ``"amide"`` or ``"hydrazide"`` (P-67.1.2.6.1: the class
                          word when every -OH is replaced by -NH2 / -NH-NH2 and the
                          amide/hydrazide is the principal group).
    ``count`` — multiplicity of the class word (``"diamide"``).

    Returns ``None`` (fail-closed) for an unknown class or out-of-range count.

    Examples::

        build_p_frn_amide_name("N,N,P,P-tetramethylphosphinic")
            -> "N,N,P,P-tetramethylphosphinic amide"
        build_p_frn_amide_name("P-phenylphosphonic", "amide", 2)
            -> "P-phenylphosphonic diamide"
    """
    if not acid_stem_word or kind not in ("amide", "hydrazide"):
        return None
    if count not in _MULT:
        return None
    return f"{acid_stem_word} {_MULT[count]}{kind}"


def build_polyacid_name(base_acid_name: str, count: int) -> Optional[str]:
    """Spell a multiplicative poly-acid (P-65.2.3 / P-67.2.1): multiplier + acid.

    ``build_polyacid_name("carbonic acid", 2) -> "dicarbonic acid"``.
    """
    if count not in _MULT or count < 2:
        return None
    return f"{_MULT[count]}{base_acid_name}"


def build_acyl_halide_name(acyl_word: str, halide_word: str, count: int) -> Optional[str]:
    """Spell an acid-halide functional-class name from an acyl-group word.

    For acids with identical replaceable groups the acyl word (``phosphoryl`` /
    ``sulfuryl`` / ``phosphorothioyl`` …) carries the structure; the halide is a
    separate class word, multiplied (P-67.1.2.5.1). ``build_acyl_halide_name(
    "phosphoryl", "chloride", 3) -> "phosphoryl trichloride"``.
    """
    if count not in _MULT:
        return None
    return f"{acyl_word} {_MULT[count]}{halide_word}"


__all__ = ["build_frn_acid_name", "build_carbonic_mono_frn", "build_polyacid_name",
           "build_acyl_halide_name",
           "build_p_frn_acid_name", "build_p_frn_acid_stem_word",
           "build_p_frn_halide_name", "build_p_frn_amide_name",
           "ACYL_PREFIX_TABLE", "acyl_prefix_for"]
