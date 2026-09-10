""" chalcogen-acid PIN spelling: italic O/S/Se/Te acid-locant cite/omit
and the (b) ``bis(thioic acid)`` multiplied-suffix enclosure.

Oracle: the Blue Book's own ``(PIN)`` worked examples (the Blue Book
``:30208`` / ``:30277``). OPSIN round-trip CANNOT decide cite-vs-omit
(InChI normalises the O/S tautomer, so both spellings round-trip to the same
structure — verified), so these assert the verbatim Blue-Book PIN string.

Derivation of the cite/omit rule: `internal notes`.
"""
import pytest

from orthonym import name_compound


# --- (b) (`:7104`) only DEFINES 'bis(thioic acid)'/'bis(dithioic acid)'
# as the multiplied two-suffix class name for two -C(=O/S)H acid groups; it
# does NOT say a chalcogen locant is omitted inside the enclosure. The bis
# form OMITs its italic locant for a DIFFERENT reason: the cited form
# `hexanebis(thioic O-acid)` is OPSIN-unparseable, and these SMILES stand in
# for a nonspecific drawing whose chalcogen location is unknown
#, `:31081`). Blue Book PINs: `hexanebis(thioic acid)`
# (`:30231`), `ethanebis(dithioic acid)` (`:30317`).
@pytest.mark.parametrize("smiles,expected", [
    # -C(=S)-OH... -C(=S)-OH chain diacid → bis(thioic acid), locant omitted
    ("OC(=S)CCCCC(O)=S", "hexanebis(thioic acid)"),   #:30231
    ("OC(=S)CCC(O)=S",   "butanebis(thioic acid)"),    #:30237
    # -CS-SH... -CS-SH homochalcogen → bis(dithioic acid) (no italic locant)
    ("S=C(S)CCCCC(=S)S", "hexanebis(dithioic acid)"),  #:30233
    ("S=C(S)C(=S)S",     "ethanebis(dithioic acid)"),  #:30317
])
def test_chain_bis_chalcogen_acid_enclosure(smiles, expected):
    assert name_compound(smiles) == expected


# --- (`:30267` / Table 4.3): ring-attached hydrazonic acid
# -C(=N-NH2)-OH -> '-carbohydrazonic acid'. No italic locant (one acid O).
# The chain form already worked (`butanehydrazonic acid`); these exercise the
# benzene ring-handler routing added in Group A. RT-safe (verified).
@pytest.mark.parametrize("smiles,expected", [
    ("NN=C(O)c1ccccc1",         "benzenecarbohydrazonic acid"),         #
    ("NN=C(O)c1ccccc1C(O)=NN",  "benzene-1,2-dicarbohydrazonic acid"),  #
])
def test_benzene_hydrazonic_acid(smiles, expected):
    assert name_compound(smiles) == expected


# --- (`:30235`) mono mixed {Se,O} chalcogen carboxylic acid:
# RESOLVED — CITE is the PIN (a review ruling, BB `:31081` /
#. The engine's `hexaneselenoic O-acid` (CITE) IS the PIN; the
# OMIT form `hexaneselenoic acid` is the unknown-chalcogen-location
# nonspecific-drawing name, which is UNSATISFIABLE from a DETERMINED SMILES.
# The FRN-A canaries (canary_functional_replacement.csv,
# tests/unit/rules/test_seniority_frn.py) assert the CITED form — correct.
# The former xfail `test_mono_selenoic_acid_omits_locant` asserted the wrong
# value (OMIT) with strict=False (a backwards alarm that would XPASS silently
# on any flip) and is deleted; the CITE behaviour is pinned above by
# `test_mono_thioic_acid_cites_locant` and the FRN-A canaries.


# --- /.2: the DETERMINED thioic {S,O} acids DO cite the italic locant
# (Blue Book marks the cited form (PIN)); these must not regress.
@pytest.mark.parametrize("smiles,expected", [
    ("CCCCCC(O)=S", "hexanethioic O-acid"),            #:30225
    ("CC(O)=S",     "ethanethioic O-acid"),            #:30291
    ("O=CS",        "methanethioic S-acid"),           #:30295
])
def test_mono_thioic_acid_cites_locant(smiles, expected):
    assert name_compound(smiles) == expected
