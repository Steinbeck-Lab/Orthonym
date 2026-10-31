"""A name built on a Chapter semisystematic or retained name is not a PIN.

 INTRODUCTION (the Blue Book): "Preferred IUPAC names (PINs) are not identified for the compounds in
this Chapter." The chapter holds carbohydrates, amino acids and peptides, cyclitols,
nucleosides and nucleotides (and natural products, lipids). A name that cites one of their
semisystematic parents or prefixes ('D-glucose', 'β-D-glucopyranosyloxy', 'L-alanine', 'adenosine', 'myo-inositol')
names the structure, but no Blue Book rule makes it the PIN, so the name is labelled below the PIN.

Out of scope here: the parents of ``data/natural_products.py`` (steroids, alkaloids, terpenes: the user's
2026-09-30 ruling keeps their labels) and lipid names. The marks are read from the name alone; none occurs in
a name the Blue Book gives as a PIN (``tests_local/unit/validation/test_p100_check_precision.py``).
"""
from __future__ import annotations

import re

from ..pin_spelling import SpellingFailure, register

#: the marks of the to semisystematic names
_P10_NAME = re.compile(
    r"pyranos|furanos|(?<!tell)uron(?:ic|ate)|inositol|osamin"     # not 'benzenetelluronic acid' (Te,
    r"|(?<![a-z])(?:[DL]|[αβ]-[DL])-[a-z]+?(?:ose|itol|ulose|onic acid|onate|aric acid|arate)\b"
    r"|(?:adenosin|guanosin|cytidin|uridin|thymidin|inosin)"
    r"|(?:glycin|alanin|valin|leucin|isoleucin|serin|threonin|cystein|methionin|prolin|phenylalanin|tyrosin"
    r"|histidin|lysin|arginin|asparagin|glutamin)(?:e|ate|amide)\b"
    r"|\btryptophan(?:ate|amide)?\b|\b(?:aspart|glutam)(?:ic acid|ate|amide)\b"
    r"|(?:glycyl|alanyl|valyl|leucyl|seryl|threonyl|cysteinyl|methionyl|prolyl|phenylalanyl|tyrosyl"
    r"|tryptophyl|histidyl|lysyl|arginyl|aspartyl|glutamyl|asparaginyl|glutaminyl)")


#: The words of the amino-acid tables (``data/amino_acids.py``: every retained name without a locant, its descriptor
#: and a final ' acid' or second word left out). A copy, so the check reads no naming code; a test keeps it equal to the
#: tables (tests/unit/validation/test_p100_semisystematic_names.py).
AMINO_ACID_TABLE_WORDS = frozenset({
    "abrine", "alanine", "alanopine", "arginamide", "arginate", "arginine", "asparagine", "aspartic",
    "beta-alanopine", "butyrine", "carnitine", "ciliatine", "creatine", "cystamine", "cysteamine", "cysteine",
    "cystinate", "cystine", "d,d-diaminopimelic", "diaminopimelic", "dibromotyrosine", "dihydroxyphenylglycine",
    "dimethyltaurate", "dopa", "glutamic", "glutamine", "glutathione", "glycine", "guvacine", "histidine",
    "homoarginamide", "homoarginate", "homocystinate", "homocystine", "homohypotaurinate", "homohypotaurine",
    "homoselenohypotaurinate", "homoselenohypotaurine", "homoselenotaurinate", "homoselenotaurine", "homotaurinate",
    "homotaurine", "hypotaurinate", "hypotaurine", "hypotaurocyamine", "isoleucine", "isolysine",
    "l,l-diaminopimelic", "leucine", "lysine", "lysopine", "methioninamine", "methionine", "methylselenocysteine",
    "nopaline", "octopine", "octopinic", "ornithine", "pantetheine", "pantothenol", "phenylalanine", "pidolate",
    "pidolic", "proline", "pyroglutamal", "pyroglutamide", "pyroglutamol", "s-methylcysteine", "sarcosine",
    "selenocystamine", "selenocysteamine", "selenocystinate", "selenocystine", "selenohypotaurinate",
    "selenohypotaurine", "selenomethionine", "selenotaurinate", "selenotaurine", "serine", "statine", "strombine",
    "taurinate", "taurine", "taurocyamine", "tauropine", "tellurocystinate", "tellurocystine", "telluromethionine",
    "threonine", "tricine", "trimethylglycine", "tryptophan", "tyrosine", "valine",
})


def _table_word_pattern(words):
    """The words and their ion and amide forms: 'pidolate', 'L-alaninium', 'lysinium(1+)', 'homocystine'. The list
    above holds only the Table 10.4 names."""
    forms = set()
    for word in words:
        forms.add(re.escape(word))
        if word.endswith("ic"):
            forms.add(re.escape(word[:-2]) + "(?:ic acid|ate)")
        elif word.endswith("ine"):
            forms.add(re.escape(word[:-1]) + "(?:e|ate|ium|amide)")
    return re.compile(r"(?<![a-z])(?:" + "|".join(sorted(forms, key=len, reverse=True)) + r")(?![a-z])")


_TABLE_NAME = _table_word_pattern(AMINO_ACID_TABLE_WORDS)


@register('P-100')
def chapter_p10_name_is_no_pin(mol, name):
    """A name that cites a to semisystematic name,:50941): no PIN is identified for it."""
    m = _P10_NAME.search(name) or _TABLE_NAME.search(name.lower())
    if m is None:
        return None
    return SpellingFailure('P-100', f"'{m.group(0)}' is a Chapter P-10 semisystematic name: "
                                    f"no PIN is identified for it")
