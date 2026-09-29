"""Breadth Job 1 (M03): the non-PIN vocabulary guard (`rules.pin_vocabulary`).

`non_pin_vocabulary` returns the first token of a name that the Blue Book names as
the non-preferred form of a PIN form. Two invariants:

* every listed non-PIN form is flagged (each case below is a form the engine built at
  best-effort, or at the PIN tier with the best-effort producers switched on);
* no Blue Book PIN / preferred prefix / preselected name is flagged -- checked
  against every name the conformance corpus extracted from the Blue Book
  (benchmarks/bb_conformance/all_pins_distinct.csv), not against engine output.
"""
import csv
from pathlib import Path

import pytest

from orthonym.assembly.memo import pin_promotion_var
from orthonym.metrics import provenance as pv
from orthonym.rules.pin_vocabulary import non_pin_vocabulary, promote_at_pin_tier

REPO = Path(__file__).resolve().parents[3]

NON_PIN = [
    # /: Hantzsch-Widman names for rings of <= 10 members
    ("6-{3-[(1E)-hexa-1,3-dien-1-yl]-1-oxacyclopropan-2-yl}hexanoic acid", "oxacycloprop"),
    ("{(2R,5R)-5-[(1R)-1-hydroxytridecyl]-1-oxacyclopentan-2-yl}", "oxacyclopent"),
    ("5-methyl-1-oxa-2-azacyclopenta-2,4-dien-3-yl", "azacyclopent"),
    ("1-oxacyclopent-3-en-2-yl", "oxacyclopent"),
    # /: benzene, phenyl
    ("14-(cyclohexa-1,3,5-trien-1-yl)tetradeca-11,13-dien-1-yl", "cyclohexa-1,3,5-trien"),
    #: an acyclic 'a' name needs four or more heterounits
    ("2-oxo-3-azabutyl", "3-azabut"),
    ("1,1-dioxo-1λ6-thia-2-azaethyl", "1λ6-thia-2-azaeth"),
    ("2-azaeth-1-yn-1-yl", "2-azaeth"),
    ("14-oxatetradecyl", "14-oxatetradec"),
    ("3-hydroxy-9-oxanon-1-en-1-yl", "9-oxanon"),
    ("1-methyl-3-oxapropyl", "3-oxaprop"),
    # 'formyl (preferred prefix)... oxomethyl'
    ("{[(2,3-dichlorophenyl)amino]-oxomethyl}amino", "oxomethyl"),
    # / retained acetic acid, acetamide
    ("N-[4-(pyrrolidin-1-yl)phenyl]ethanamide", "ethanamid"),
    ("(adamantan-1-yl)methyl 2-chloroethanoate", "ethanoat"),
    ("2-[(4E)-octa-4,7-dienamido]ethanoic acid", "ethanoic acid"),
    #: -NH-CO-R is '-amido'
    ("2-acetylamino-3-(acetyloxy)-3-oxopropyl", "acetylamino"),
    ("[(2-methylpropanoyl)amino]", "panoyl)amino"),
    # /: esters are functional class unless a senior group
    ("(2R)-3-(docosanoyloxy)-2-(tetradecanoyloxy)propane", "sanoyloxy"),
    # 'propan-1-yl... propyl (preferred prefix)'
    ("3-(cyclopropan-1-yl)", "cyclopropan-1-yl"),
    ("N-[2-(pyridin-2-yl)ethan-1-yl]", "ethan-1-yl"),
    # /: ortho-fused systems take fusion names
    ("2-oxabicyclo[4.4.0]deca-1(10),3,6,8-tetraen-3-yl", "bicyclo[4.4.0]"),
    ("7-azabicyclo[4.3.0]nona-1,3,5,8-tetraen-9-yl", "bicyclo[4.3.0]"),
    ("2,15-dimethyltetracyclo[8.7.0.0^2,7.0^11,15]heptadec-7-en-5-yl",
     "tetracyclo[8.7.0.0^2,7.0^11,15]"),
    #: adamantane is retained for the PIN
    ("tricyclo[3.3.1.1^3,7]decane", "tricyclo[3.3.1.1^3,7]dec"),
    ("[tricyclo[3.3.1.1^3,7]decane-1-carboxamido]acetate", "tricyclo[3.3.1.1^3,7]dec"),
    #: methoxy... butoxy, phenoxy are contracted
    ("1-[(1R)-1-{[(2-benzylphenyl)oxy]methyl}ethyl]piperidine", "phenyl)oxy"),
    ("3-({2-[(2-methoxyphenyl)oxy]ethyl}amino)propan-2-ol", "phenyl)oxy"),
    ("[(2-methylpropyl)oxy]", "propyl)oxy"),
    # /: longest chain of a substituent prefix
    ("2-({1-[(adamantan-1-yl)methyl]ethyl}amino)ethan-1-ol", "methyl]ethyl"),
    ("4-(1-methylethyl)phenyl", "methylethyl"),
    ("1-(hydroxymethyl)ethyl", "methyl)ethyl"),
    #: hydroperoxy
    ("(2R)-2-dioxidanyl-3-(phosphonooxy)propyl", "dioxidanyl"),
    #: propanoyl, not 1-oxopropyl
    ("(2S)-2-[(3-carboxylato-1-oxopropyl)amino]-6-oxoheptanedioate", "1-oxopropyl"),
    #: compound prefixes are enclosed
    ("(1R)-2-cyclopropylamino-2-oxo-1-phenylethyl 2-cyclopropylquinoline-4-carboxylate",
     "2-cyclopropylamino"),
    ("S-8-amino-8-oxo-3-sulfanyloctyl 2-methylbutanethioate", "S-8-"),
    #: all locants once one is needed
    ("4,4-difluoro-N-[(1S)-1-(4-hydroxyphenyl)propyl]cyclohexanecarboxamide",
     "cyclohexanecarboxamide"),
    ("methyl 4-methylcyclohexanecarboxylate", "cyclohexanecarboxylate"),
    # /: no suffix, so no suffix-expressible prefix
    ("(2-acetamidopropyl)benzene", "amido"),
    ("1-(3-hydroxy-2,2,6-trimethyl-3,4-dihydro-2H-1-benzopyran-4-yl)-2-oxopyrrolidine",
     "hydroxy"),
    ("(4S)-4-ethyl-2-[(pyridin-3-yl)amino]-4,5-dihydro-1,3-thiazole", "amino"),
    ("(3S,4S)-3,4-dibromo-3-methyl-1,1-dioxothiolane", "1,1-dioxo"),
    # +: an ester is a prefix only beside a SENIOR suffix
    ("4-(acetyloxy)phenol", "acetyloxy"),
    ("1-(propanoyloxy)pyrrolidine-2,5-dione", "panoyloxy"),
    ("2-(acetylsulfanyl)ethan-1-ol", "acetylsulfanyl"),
    ("3-(cyclohexanesulfinylsulfanyl)propanenitrile", "esulfinylsulfanyl"),
    ("2-[(methanesulfonyl)oxy]ethan-1-ol", "esulfonyl)oxy"),
    ("(2R)-3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}-2-(octadecanoyloxy)propan-1-ol",
     "canoyloxy"),
    #: a parent that needs indicated hydrogen keeps it on a ring ketone
    ("5-(5-hydroxy-1H-indol-3-yl)-3-(1H-indol-3-yl)pyrrol-2-one", "pyrrol-2-one"),
    ("6-[(2S)-2-hydroxypropyl]-4-hydroxy-3-methylpyran-2-one", "pyran-2-one"),
    ("indole-2,3-dione", "indole-2,3-dione"),
    #: a locant-bearing substituent after an N locant is enclosed
    ("N-5,6,7,8-tetrahydronaphthalen-2-ylnaphthalen-2-amine", "N-5,6,7,8-"),
    #: 'bis' for a compound prefix
    ("3-[2,3-di(carboxymethyl)naphthalen-1-yl]propanoic acid", "di(carboxymethyl)"),
    #: identical acetic acid parents are multiplied
    ("[3-(carboxymethyl)naphthalen-2-yl]acetic acid", "carboxymethyl"),
    #: Hantzsch-Widman for P / B rings too
    ("1-(2-oxo-1-phosphacycloheptan-3-yl)propan-1-one", "phosphacyclohept"),
    ("1-boracyclohexan-1-yl acetate", "boracyclohex"),
    # /: a bridged fused name beats the von Baeyer name
    ("tricyclo[5.2.1.0^2,6]decane", "tricyclo[5.2.1.0^2,6]"),
    ("tricyclo[12.3.1.0^5,10]octadecane", "tricyclo[12.3.1.0^5,10]"),
    ("4-ethyl-5-hydroxy-1,4,6,12-tetramethyl-10-oxotricyclo[5.4.3.0^7,11]tetradecan-2-yl",
     "tricyclo[5.4.3.0^7,11]"),
    # (a): a hyphen separates a locant from a word
    ("3-hydroxy-2,2-dimethyl3,4-dihydro-2H-1-benzopyran-6-yl", "l3"),
    ("5-(5-hydroxy1H-indol-3-yl)", "y1"),
    #: one locant per descriptor of a set
    ("(1S,1S,2S,4R)-N-[1-(bicyclo[2.2.1]heptan-2-yl)ethyl]thiourea", "(1S,1S,2S,4R)"),
]

PIN_FORMS = [
    "7-oxabicyclo[4.1.0]hept-3-en-3-yl",           # three-membered ring: von Baeyer PIN
    "2,6-dioxabicyclo[3.1.1]heptan-4-yl",
    "bicyclo[4.2.0]octa-1,3,5,7-tetraene",          # the Blue Book
    "2,6-dimethyltricyclo[3.2.0.0^2,6]heptane-1-carboxylate",  # bridged, not fused
    "bicyclo[2.2.1]heptan-2-yl",                    # no fused pair
    "tricyclo[2.2.1.0^2,6]heptane",                 # fused pair only with a 3-ring
    "3,6,9,12-tetraoxatetradecanedioic acid",       # four heterounits
    "1-oxacyclotridecane",                          # > 10 ring members
    "oxan-2-yl",
    "N-(propan-2-yl)acetamide",
    "cyclohexan-1-yl-2-ylidene",
    "3-(adamantan-1-yl)propan-2-yl",
    "(4-methylcyclohex-3-en-1-yl)acetic acid",
    "(1R,2S)-2-methylcyclohexan-1-ol",
    "carbamothioylamino",
    "4-formamidobenzoic acid",
    "oxomethylidene",
    "3-(sulfooxy)propanoic acid",
    "propane-1,2,3-triyl triacetate",
    "3-(acetyloxy)propanoic acid",
    "N-hydroxycyclohexanecarboxamide",               # the Blue Book
    "3-(2-iminopropyl)cyclohexane-1-carboxylic acid",  #:26549
    "2-bromo-2-(bromomethyl)butyl",                  #:15787
    "(butan-2-yl)oxy",                               #:27639
    "1-oxopropan-2-yl",
    "1-hydroxypropan-2-ylidene",                     #:16349
    "(O-2H,18O)acetic acid",                         #:43808
    "S-(2-cyanoethyl) ethanethioate",
    "4-(aminomethyl)aniline",
    "1,3-dioxolane",
    "1,5-dioxocane",
    "N-methylethenamine",
    "furan-2,5-dione",
    "isocyanobenzene",
    "2-(acetyloxy)benzoic acid",                     # the acid is senior
    "2-hydroxyethyl acetate",
    "2-(acetyloxy)-N,N,N-trimethylethan-1-aminium",  # a cation is senior
    "1-[(acetylsulfanyl)peroxy]propan-1-one",        #:39499
    "2H-pyrrol-2-one",
    "1H-pyrrole-2,5-dione",
    "4-hydroxy-6-methyl-2H-pyran-2-one",
    "5,6-dihydro-2H-pyran-2-one",
    "1,3-dihydro-2H-indol-2-one",
    "4H-pyrido[1,2-a]pyrimidin-4-one",
    "1H,3H-benzo[de][2]benzopyran-1,3-dione",         #:32498
    "pyridin-2(1H)-one",
    "1,2,5,6-tetrasilacyclooct-3-en-7-yne",          #:21091, a triple bond
    "4-(benzenesulfonyl)phenol",                     # a sulfone, not an ester
    "N,N-di(propan-2-yl)acetamide",
    "N-(2-hydroxyethyl)acetamide",
]

# Rows of the extracted corpus that are extraction artefacts, not BB strings: the
# 'propan-1-yl propyl (preferred prefix)' row lists the non-preferred form first
# (the Blue Book), and the boranuide row lost the braces of
# 'trimethyl{2-[methyldi(phenyl)phosphaniumyl]ethen-1-yl}boranuide (PIN)' (:42468).
_CORPUS_ARTEFACTS = {
    "propan-1-yl propyl",
    "1-methylethylidene propan-2-ylidene",  #:16186, non-preferred form listed first
    # braces lost in extraction: the Blue Book '...-*N*-{2-[(2-ethoxyethyl)
    # (methyl)sulfaniumyl]ethyl}-...' and:33574 '...-*N*-{2-[(2-hydroxyethyl)amino]
    # ethyl}butanamide (PIN)'
    "2-ethoxy-N-2-[(2-ethoxyethyl)(methyl)sulfaniumyl]ethyl-N,N-dimethylethanaminium",
    "4-[(2-hydroxyethyl)amino]-N-2-[(2-hydroxyethyl)amino]ethylbutanamide",
    "trimethyl2-[methyldi(phenyl)phosphaniumyl]ethen-1-ylboranuide",
}


# A Blue Book row that contradicts the Blue Book's own rule and every other example
# of it: (the Blue Book, "then all locants must be cited") and
# (c) (:2913, locant '1' omitted only in MONOsubstituted rings), with
# twenty '...cyclohexane-1-carboxylic acid (PIN)' / '-1-carboxamide (PIN)' rows
# (:6134,:26549,:31535,:48077,...). Only:30363 writes a substituted ring
# without the locant.
_BB_INCONSISTENT = {
    "4-[(hydroxysulfanyl)carbonyl]cyclohexanecarboxylic acid",
}


@pytest.mark.parametrize("name,token", NON_PIN)
def test_non_pin_form_is_flagged(name, token):
    assert non_pin_vocabulary(name) == token


@pytest.mark.parametrize("name", PIN_FORMS)
def test_pin_form_is_not_flagged(name):
    assert non_pin_vocabulary(name) is None


def test_no_blue_book_pin_is_flagged():
    path = REPO / "benchmarks" / "bb_conformance" / "all_pins_distinct.csv"
    rows = list(csv.DictReader(path.open()))
    assert len(rows) > 3000
    flagged = [(r["bb_line"], r["bb_name"], non_pin_vocabulary(r["bb_name"]))
               for r in rows
               if r["bb_name"] not in _CORPUS_ARTEFACTS | _BB_INCONSISTENT
               and non_pin_vocabulary(r["bb_name"])]
    assert flagged == []


def test_promotion_is_inactive_outside_the_pin_rerun():
    calls = []
    assert promote_at_pin_tier(lambda: calls.append(1) or "phenyl") is None
    assert calls == []


def test_promotion_keeps_only_pin_vocabulary_and_rolls_back_provenance():
    tok = pin_promotion_var.set(True)
    try:
        pv.clear_provenance()
        assert promote_at_pin_tier(lambda: "(4-methoxyphenyl)methyl") == "(4-methoxyphenyl)methyl"

        def _non_pin():
            pv.record_general_ring_prefix()
            return "1-oxacyclopentan-2-yl"
        assert promote_at_pin_tier(_non_pin) is None
        assert pv.get_provenance()["general_ring_prefix"] is False

        def _general_prefix():
            pv.record_general_ring_prefix()
            return "bicyclo[2.2.2]octan-2-yl"
        assert promote_at_pin_tier(_general_prefix) is None
        assert pv.get_provenance()["general_ring_prefix"] is False
    finally:
        pin_promotion_var.reset(tok)
        pv.clear_provenance()


def test_multiplicative_candidate():
    from rdkit import Chem

    from orthonym.rules.pin_vocabulary import multiplicative_candidate as mc
    # identical ring systems carrying the principal group, or the senior ring system
    # repeated with no principal group: a multiplicative PIN may apply
    for smi in ("N#Cc1ccccc1Cc1ccccc1C#N", "ClC(Cc1ccccc1)c1ccccc1",
                "O=C(O)C1CCCCC1COCCOCC1CCCCC1C(=O)O"):
        assert mc(Chem.MolFromSmiles(smi)), smi
    # a repeated ring that is only a substituent of a senior parent, or a principal
    # group on the linking chain, or a single ring system: substitutive
    for smi in ("COc1ccc(Cc2ncc(Cc3ccc(OC)cc3)c(C)n2)cc1", "OC(c1ccc(Cl)cc1)c1ccc(Cl)cc1",
                "NCc1csc(-c2cccs2)n1", "CCO"):
        assert not mc(Chem.MolFromSmiles(smi)), smi
