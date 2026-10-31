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


@register('P-100')
def chapter_p10_name_is_no_pin(mol, name):
    """A name that cites a to semisystematic name,:50941): no PIN is identified for it."""
    m = _P10_NAME.search(name)
    if m is None:
        return None
    return SpellingFailure('P-100', f"'{m.group(0)}' is a Chapter P-10 semisystematic name: "
                                    f"no PIN is identified for it")
