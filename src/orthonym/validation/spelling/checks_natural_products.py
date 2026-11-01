"""A name built on a Chapter semisystematic or retained name is not a PIN.

 INTRODUCTION (the Blue Book): "Preferred IUPAC names (PINs) are not identified for the compounds in
this Chapter." The chapter holds carbohydrates, amino acids and peptides, cyclitols,
nucleosides and nucleotides (and natural products, lipids). A name that cites one of their
semisystematic parents or prefixes ('D-glucose', 'β-D-glucopyranosyloxy', 'L-alanine', 'adenosine', 'myo-inositol')
names the structure, but no Blue Book rule makes it the PIN, so the name is labelled below the PIN.

The names of ``data/natural_products.py`` are marked too (user decision 2026-10-31, which replaces the 2026-09-30
ruling that kept their labels): the fundamental parent structures (``### ****...`` lists,:51409
steroids,:51413 terpenes,:51419 miscellaneous) and their stems ('5β-cholan-24-oate', 'androst-4-ene-3,17-dione'),
and the trivial names of the same tables ('cholesterol', 'lysergic acid', 'alpha-terpineol'), which no rule of the
book makes a PIN. Out of scope: lipid names. The marks are read from the name alone; none occurs in a name the
Blue Book gives as a PIN (``tests_local/unit/validation/test_p100_check_precision.py``).
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

#: The names of the natural-product tables (``data/natural_products.py``: NATURAL_PRODUCT_SCAFFOLDS,
#: NATURAL_PRODUCT_DERIVATIVES, NAME_EXACT_NP_PARENTS), lower case, a leading E/Z descriptor left out. A copy, as above
#: (tests/unit/validation/test_p100_semisystematic_names.py keeps it equal to the tables).
NATURAL_PRODUCT_TABLE_WORDS = frozenset({
    "1-carbapen-1-em", "21h-biline", "4-terpineol", "abietane", "aconitane", "ajmalan", "alpha-pinene",
    "alpha-terpinene", "alpha-terpineol", "ambrosane", "androstadienone", "androstane", "androstanediol",
    "androstenediol", "androstenedione", "androstenol", "androstenone", "aporphine", "aristolane", "aspidospermidine",
    "atisane", "berbine", "beta-carotene", "beta-terpinene", "beta-terpineol", "beyerane", "bornane", "bufadienolide",
    "bufanolide", "cadinane", "campestane", "campestanol", "camphor", "cardanolide", "cardenolide", "ceph-2-em",
    "ceph-3-em", "cepham", "cevane", "cholane", "cholestane", "cholesterol", "cinchonan", "codeine", "codeinone",
    "conanine", "corrin", "corynan", "corynoxan", "cucurbitane", "dammarane", "delta-terpinene", "diamorphine",
    "dihydrocodeine", "dihydroheroin", "dihydroisolysergamide", "dihydroisolysergic acid", "dihydroisolysergol",
    "dihydrolysergamide", "dihydrolysergic acid", "dihydrolysergol", "dihydromorphine", "drimane", "emetan",
    "eremophilane", "ergoline", "ergosta-7,24(28)-dien-3-ol", "ergostane", "erythrinan", "estrane", "estratetraenol",
    "eudesmane", "furostan", "galanthan", "gamma-terpinene", "gamma-terpineol", "gammacerane", "germacrane", "gibbane",
    "gonane", "gorgostane", "grayanotoxane", "guaiane", "hasubanan", "hetisan", "himachalane", "hopane",
    "hydrocodone", "hydromorphone", "ibogamine", "isolysergamide", "isolysergic acid", "isolysergol", "kaurane",
    "kopsan", "labdane", "lanostane", "limonene", "lupane", "lycopene", "lycopodane", "lysergamide", "lysergic acid",
    "lysergol", "menthane", "morphinan", "morphine", "morphinone", "oleanane", "ophiobolane", "oxycodone", "pen-2-em",
    "penam", "picrasane", "pimarane", "pinane", "podocarpane", "poriferastane", "porphyrin", "pregnane",
    "progesterone", "prostane", "protostane", "rosane", "sarpagan", "solanidane", "sparteine", "sphing-4-enine",
    "spirostan", "stigmastane", "strychnidine", "taxane", "thromboxane", "trichothecane", "tropane", "tubulosan",
    "ursane", "vincane", "yohimban",
})

#: The stems the parents take before an ending or a locant ('cholest-5-ene', 'androsta-1,4-diene',
#: '5β-cholan-24-oic acid', 'trop-6-en-3-ol', '17-methylmorphin-7-ene'): the ``stem`` of every scaffold of
#: NATURAL_PRODUCT_SCAFFOLDS and the four steroid parents of the derivatives table (a test keeps it equal to the table).
PARENT_STEMS = ("androst", "estr", "pregn", "cholest", "chol", "ergost", "campest", "stigmast", "gon", "morphin",
                "trop", "cinchon", "aporphin", "ergolin", "aconit", "berbin", "ajmal",
                "gorgost", "poriferast", "spirost", "furost")

_NP_NAME = re.compile(
    r"(?:(?:" + "|".join(sorted(PARENT_STEMS, key=len, reverse=True)) + r")(?:a-|an(?:e|-|\b)|-\d)"
    r"|(?:morphinan|tropan|cinchonan|aporphin|ergolin|aconitan|berbin|ajmalan)(?:e|-|\b)"
    r"|(?:" + "|".join(re.escape(w) for w in sorted(NATURAL_PRODUCT_TABLE_WORDS, key=len, reverse=True)) + r")"
    r"(?![a-z]))")


@register('P-100')
def chapter_p10_name_is_no_pin(mol, name):
    """A name that cites a to semisystematic name,:50941): no PIN is identified for it."""
    low = name.lower()
    m = _P10_NAME.search(name) or _TABLE_NAME.search(low) or _NP_NAME.search(low)
    if m is None:
        return None
    return SpellingFailure('P-100', f"'{m.group(0)}' is a Chapter P-10 semisystematic name: "
                                    f"no PIN is identified for it")
