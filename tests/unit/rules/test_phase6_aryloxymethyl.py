import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


@pytest.mark.opsin_gate
def test_nitrophenoxymethyl_oxirane_names(namer):
    # PIN = "[(4-nitrophenoxy)methyl]oxirane" (RT-verified below).
    #
    # Ring locant: OMITTED. BB P-13.1 table row 7 (verbatim, see
    # test_p14_3_4_task3b_baked_locants.py) gives `phenyloxirane (PIN)` for a
    # monosubstituted oxirane -- the two ring CH2 positions are one orbit
    # (`l3_one_kind_of_substitutable_h`), independent of what decorates the
    # substituent, so no "2-" is cited here either.
    #
    # Brackets: NESTED, not collapsed. `4-nitrophenoxy` (P-63.2.2.2 keeps the
    # retained CONTRACTED "phenoxy" spelling even when the ring is substituted
    # -- BB verbatim `(4-chlorophenoxy)benzene`, `(2-nitrophenoxy)borane`) is
    # itself a LOCANT-BEARING (compound) sub-component of the larger
    # substituent prefix `{oxy}methyl`, so per P-16.3.3/P-16.5 it takes its own
    # enclosing marks BEFORE concatenating "methyl", exactly parallel to the
    # analogous (X-phenyl)+methyl/methoxy sub-component pattern the Blue Book
    # prints verbatim: `[(3-chlorophenyl)methyl]benzene`,
    # `2-[(4-bromophenyl)methyl]pyridine`, `4-[(4-hydroxyphenyl)methyl]phenol`.
    # The retained contraction changes the SPELLING of the atomic unit
    # ("phenoxy" vs "(phenyl)oxy"), not whether a decorated, LOCANTED version
    # of it needs its own marks as a sub-component. The outer citation on
    # oxirane then escalates (-> [ over the inner marks. The decision is
    # keyed on `starts_with_locant` (does the oxy prefix CITE a locant), not
    # on `is_complex_substituent`/`enclose_if_compound` -- those flag
    # 'benzyloxy'/'cyclohexyloxy' as compound two-morpheme prefixes even
    # though they carry no locant, which over-nests and regresses the gold
    # row covered by test_benzyloxy_and_cyclohexyloxy_stay_bare below.
    smi = "O=[N+]([O-])c1ccc(OCC2CO2)cc1"
    name = namer.name(smi)
    assert name == "[(4-nitrophenoxy)methyl]oxirane", name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # Unsubstituted phenoxy carries no locant of its own -> stays bare, single
    # parens (no inner sub-component to nest).
    ("c1ccc(OCC2CO2)cc1", "(phenoxymethyl)oxirane"),
    # PRE-EXISTING non-PIN collapse, corrected here (change-asserted-value):
    # "4-methylphenoxy" is exactly as locant-bearing/compound as
    # "4-nitrophenoxy" above and takes the identical nested-bracket treatment
    # -- RT-verified below. The old collapsed "(4-methylphenoxymethyl)oxirane"
    # was never a valid PIN spelling; it just happened to round-trip.
    ("Cc1ccc(OCC2CO2)cc1", "[(4-methylphenoxy)methyl]oxirane"),
])
def test_plain_aryloxymethyl_unchanged(namer, smi, expected):
    smi_ = smi
    name = namer.name(smi_)
    assert name == expected, name
    assert _full_rt(smi_, name), name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # PROTECT-ROW REGRESSION LOCK. A first fix used `enclose_if_compound`, which
    # flags 'benzyloxy'/'cyclohexyloxy' as compound two-morpheme prefixes and
    # over-nested these — regressing the gold PIN `4-(benzyloxymethyl)phenol`
    # (benchmarks/the gold set, category characteristic_groups) to a wrong
    # `4-[(benzyloxy)methyl]phenol`. A retained oxy prefix that cites NO locant
    # of its own stays BARE (single parens, no inner nest) — the enclosure
    # decision is keyed on `starts_with_locant`, not on compound-ness.
    ("Oc1ccc(COCc2ccccc2)cc1", "4-(benzyloxymethyl)phenol"),
    ("Clc1ccc(COC2CCCCC2)cc1", "1-chloro-4-(cyclohexyloxymethyl)benzene"),
])
def test_benzyloxy_and_cyclohexyloxy_stay_bare(namer, smi, expected):
    name = namer.name(smi)
    assert name == expected, name
    assert _full_rt(smi, name), name
