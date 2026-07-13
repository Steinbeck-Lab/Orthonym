"""Shared functional-replacement (FRN) acid-name engine — P-67.1.2 / P-65.2.1.2.

ONE name-builder for the functional-replacement / functional-class derivatives of
the retained acid *parents* (carbonic, carbamic, phosphoric, sulfuric, …), used by
BOTH the carbon acids (P-65.2 carbonic family) and the non-carbon oxoacids (P-67).
Per the v23 cross-cutting design (§7, FRN-infix owner = Phase 9) the construction
logic lives here once, not duplicated per family.

This module is PURE string assembly: it does NOT perceive structure. The caller
(``rules.inorganic_acids`` / future P-67 non-carbon adapters) decides *which*
replacement a given structure represents — keyed on the exact canonical SMILES,
fail-closed — and asks this engine to spell the name. Keeping perception out of
the builder is what makes it safely reusable and free of false positives.

Coverage in v23 Phase 9 = the OPSIN-round-trippable plain forms:
  * chalcogen / peroxo infix replacement on a retained acid  (``carbonoperoxoic``,
    ``carbonodithioic``, ``carbonotrithioic``)
  * the =O -> =NH imido replacement                          (``carbonimidic``,
    ``carbamimidic``)
  * multiplicative poly-acids                                (``dicarbonic``,
    ``tricarbonic``)
  * acyl-halide functional-class words                       (``phosphoryl
    trichloride``, ``sulfuryl dichloride``)
The italic O/S/Se tautomer-locant word-forms (``carbonothioic S-acid``) are
DEFERRED to Phase 19 — OPSIN rejects the word-form, so they require name-exact
gold rather than round-trip validation (V23 plan §6, Phase 19).
"""
import re
from typing import Dict, List, Optional, Tuple

# Numerical multiplying prefixes (P-14.2). Index = count.
_MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}

# --- W3-P10 (P-67.1.2.3.2 / P-67.1.2.4): class infixes for the mononuclear-P
# oxoacid functional-replacement engine. Each combining form carries a trailing
# linking ``o`` that is elided before a vowel (P-67.1.2.3.5). The chalcogen /
# peroxo infixes (thio/seleno/telluro/peroxo) do NOT belong here — they never
# elide their ``o`` and are spelled by :func:`build_frn_acid_name`. Keyed by the
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

    ``front_prefix``  — already-assembled detachable-prefix string cited in front
                        (organyl on P: ``"methyl"`` / ``"phenyl"``; or the
                        N-locant amido substituents: ``"N,N-dimethyl"``). ``""``
                        for the bare parent.
    ``parent_stem``   — parent-acid stem WITHOUT its linking vowel: ``"phosphor"``
                        (phosphoric), ``"phosphon"`` (phosphonic), ``"phosphin"``
                        (phosphinic), ``"arsor"``/``"arson"`` etc.
    ``infix_counts``  — ``{combining-form: multiplicity}`` for the class infixes
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
    parts = [parent_stem + "o"]                       # phosphor -> phosphoro
    for infix in sorted(infix_counts):                # alphabetical (P-67.1.2.3.5)
        if infix not in _CLASS_INFIX:
            return None
        cnt = infix_counts[infix]
        if cnt not in _MULT:
            return None
        parts.append(f"{_MULT[cnt]}{infix}")
    stem_word = _elide_o_before_vowel("".join(parts) + "ic")
    return f"{front_prefix}{stem_word} acid"

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
        build_frn_acid_name("carbon", "thio", 2)   -> "carbonodithioic acid"
        build_frn_acid_name("carbon", "thio", 3)   -> "carbonotrithioic acid"
        build_frn_acid_name("carbon", "imido", 1)  -> "carbonimidic acid"
        build_frn_acid_name("carbam", "imido", 1)  -> "carbamimidic acid"
    """
    if count not in _MULT:
        return None
    if infix in _CHALCOGEN_INFIX:
        # carbon + o + [di|tri] + thio + ic acid
        return f"{base_stem}o{_MULT[count]}{infix}ic acid"
    if infix in _IMIDO_INFIX:
        # carbon + imid + ic acid   (linking o elided; multiplier rare, supported)
        return f"{base_stem}{_MULT[count]}{_IMIDO_INFIX[infix]}ic acid"
    return None


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


__all__ = ["build_frn_acid_name", "build_polyacid_name", "build_acyl_halide_name",
           "build_p_frn_acid_name"]
