"""a phase: catalog byte-identical lock verification.

Per 149-internal notes: when match_fused_heterocycle_core returns
non-None, NOTHING in the call graph changes. Cataloged compounds flow
through namer._build_ring_info_for_parent_selection Branch 1 (existing);
their iupac_locants come from the catalog's iupac_locants dict; a phase's
Branch 6.5 never fires for cataloged compounds.

This test parametrizes over every entry in FUSED_HETEROCYCLE_DATA and
asserts name_compound(smiles) == post-148.2-baseline-name byte-identical.
The baseline name table is FROZEN at Plan 02 capture time per a phase
determinism doctrine (captured by Task 02-00 BEFORE any Plan 02 source
modifications landed).

Source: 149-internal notes (additive, byte-identical-on-catalog).
Source: 148-internal notes / byte-identical preservation precedent.
Source: acceptance gate (G6 in internal notes).
"""
import pytest

from orthonym import name_compound
from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA


# Frozen post-148.2 baseline name table. Keys are canonical SMILES from
# FUSED_HETEROCYCLE_DATA; values are the byte-identical name_compound
# output captured by Task 02-00 against the post-148.2 commit BEFORE
# Plan 02 source modifications (//) landed. Captured once,
# committed as a literal — NOT re-derived at test-run time per Phase
# 145.2 determinism doctrine.
#
# Capture artifact: tests/unit/data/_post_148_2_baseline.json (committed
# alongside Plan 02 — the JSON is the immutable record; this dict
# literal is the test-time consumer per RESEARCH §"FROZEN baseline
# dict-literal" line 565).
POST_148_2_BASELINE_NAMES = {
    'C1=CC=C2C=CC=CC=C2C=C1': 'heptalene',
    'C1=CCN2C=CC=CC2=C1': '4H-quinolizine',
    'C1=COc2ccccc2C1': '4H-1-benzopyran',
    'C1=Cc2c3c(ccc2=C1)=CC=C3': 'as-indacene',
    'C1=Cc2cc3c(cc2=C1)C=CC=3': 's-indacene',
    'C1=Cc2cccc3cccc(c23)C1': '1H-phenalene',
    'C1=Cc2cccc3cccc1c23': 'acenaphthylene',
    'C1=Cc2ccccc2C1': '1H-indene',
    'C1=CSc2ccccc2C1': '4H-1-benzothiopyran',
    'C1=Cc2ccccc2CO1': '1H-2-benzopyran',
    'C1=Cc2ccccc2OC1': '2H-1-benzopyran',
    'C1=Cc2ccccc2SC1': '2H-1-benzothiopyran',
    # fix a performance pass (wp6-tests), change-asserted-value (was 'pyrrolizine', which the
    # engine still ships at pin_verified): Table 2.8 '(21) pyrrolizine (1H-isomer shown;
    # the PIN is 1H-pyrrolizine)' (the Blue Book). OPSIN 2.9.0 full-InChIKey exact.
    # Strict xfail below until the catalog spells the indicated hydrogen.
    'C1=Cn2cccc2C1': '1H-pyrrolizine',
    'C1=NCc2ccccc2C1': '1,4-dihydroisoquinoline',
    'C1=Nc2cccc3cccc(c23)N1': '1H-perimidine',
    'C1=Nc2ccccc2C1': '3H-indole',
    # fix a performance pass (wp6-tests), change-asserted-value (were 'pyrrolizidine' and
    # 'indolizidine'; 0 Blue Book hits, labelled below pin_verified since wp5): the
    # fully saturated parents take hydro prefixes, the Blue Book)
    # on the PIN parents '1H-pyrrolizine' (:11628) and 'indolizine (PIN)' (:11709).
    # OPSIN 2.9.0 full-InChIKey exact. Strict xfail below until the PINs are built
    # (the same class as the quinolizidine row).
    'C1CC2CCCN2C1': 'hexahydro-1H-pyrrolizine',
    'C1CCN2CCCC2C1': 'octahydroindolizine',
    # fix a performance pass (wp5): the frozen 'decahydroisoquinoline' was a different molecule
    # (the row was deleted on purpose, retained_names.py). The PIN is 'octahydro-2H-quinolizine'
    # (quinolizine the Blue Book-11584; hydro prefixes the Blue Book; OPSIN 2.9.0 full-InChIKey
    # exact); the catalog still says 'quinolizidine', now labelled below pin_verified. Strict
    # xfail below until the PIN is built.
    'C1CCN2CCCCC2C1': 'octahydro-2H-quinolizine',
    'Nc1nc(=O)c2[nH]cnc2[nH]1': 'guanine',
    'Nc1nc2[nH]cnc2c(=O)[nH]1': 'guanine',
    'Nc1ncnc2[nH]cnc12': 'adenine',
    'Nc1ncnc2nc[nH]c12': 'adenine',
    'O=C1c2ccccc2-c2ccccc21': '9H-fluoren-9-one',
    'O=c1[nH]c(=O)c2[nH]cnc2[nH]1': 'xanthine',
    'O=c1[nH]c(=O)c2nc3ccccc3nc2[nH]1': 'benzo[g]pteridine-2,4(1H,3H)-dione',
    'O=c1[nH]c(=O)c2nc[nH]c2[nH]1': 'xanthine',
    'O=c1[nH]c2ccccc2c2ccccc12': 'phenanthridin-6(5H)-one',
    'O=c1[nH]cnc2[nH]cnc12': 'hypoxanthine',
    'O=c1[nH]cnc2nc[nH]c12': 'hypoxanthine',
    'O=c1c2ccccc2[nH]c2ccccc12': 'acridin-9(10H)-one',
    'O=c1c2ccccc2oc2ccccc12': '9H-xanthen-9-one',
    'O=c1c2ccccc2sc2ccccc12': '9H-thioxanthen-9-one',
    'O=c1ccc2ccccc2o1': '2H-1-benzopyran-2-one',
    'c1cc2[nH]ccc2cn1': '1H-pyrrolo[3,2-c]pyridine',
    'c1cc2[nH]ncc2cn1': '1H-pyrazolo[4,3-c]pyridine',
    'c1cc2cc[nH]c2cn1': '1H-pyrrolo[2,3-c]pyridine',
    'c1cc2ccc3cccc4ccc(c1)c2c34': 'pyrene',
    'c1cc2ccncc2cn1': '2,7-naphthyridine',
    'c1cc2ccncn2c1': 'pyrrolo[1,2-c]pyrimidine',
    'c1cc2ccsc2cn1': 'thieno[2,3-c]pyridine',
    'c1cc2ccsc2nn1': 'thieno[2,3-c]pyridazine',
    'c1cc2ccsc2s1': 'thieno[2,3-b]thiophene',
    'c1cc2cnccn2c1': 'pyrrolo[1,2-a]pyrazine',
    'c1cc2nc[nH]cc-2n1': '3H-pyrrolo[3,2-d]pyrimidine',
    'c1cc2nccnc2cn1': 'pyrido[3,4-b]pyrazine',
    'c1cc2nccnc2nn1': 'pyrazino[2,3-c]pyridazine',
    'c1cc2ncncc2cn1': 'pyrido[4,3-d]pyrimidine',
    'c1cc2ncoc2cn1': 'oxazolo[5,4-c]pyridine',
    'c1cc2ncsc2cn1': 'thiazolo[5,4-c]pyridine',
    'c1cc2occc2cn1': 'furo[3,2-c]pyridine',
    'c1cc2sccc2cn1': 'thieno[3,2-c]pyridine',
    'c1cc2sccc2s1': 'thieno[3,2-b]thiophene',
    'c1ccc2[nH]ccc2c1': '1H-indole',
    'c1ccc2[nH]cnc2c1': '1H-benzimidazole',
    'c1ccc2[nH]ncc2c1': '1H-indazole',
    'c1ccc2[nH]nnc2c1': '1H-benzotriazole',
    'c1ccc2[se]cnc2c1': '1,3-benzoselenazole',
    'c1ccc2c(c1)-c1ccccc1-2': 'biphenylene',
    # Phase C: (the Blue Book) names the retained forms 'indane' /
    # 'indoline' / 'isoindoline' verbatim as NOT preferred IUPAC names; the Blue Book /
    #:16992 /:16999 print the PINs. The baseline is re-frozen on the PIN spelling.
    'c1ccc2c(c1)CCC2': '2,3-dihydro-1H-indene',
    'c1ccc2c(c1)CCCN2': '1,2,3,4-tetrahydroquinoline',
    'c1ccc2c(c1)CCCO2': '3,4-dihydro-2H-1-benzopyran',
    'c1ccc2c(c1)CCCS2': '3,4-dihydro-2H-1-benzothiopyran',
    'c1ccc2c(c1)CCN2': '2,3-dihydro-1H-indole',
    'c1ccc2c(c1)CCNC2': '1,2,3,4-tetrahydroisoquinoline',
    'c1ccc2c(c1)CCO2': '2,3-dihydro-1-benzofuran',
    'c1ccc2c(c1)CCOC2': '3,4-dihydro-1H-2-benzopyran',
    'c1ccc2c(c1)CCS2': '2,3-dihydro-1-benzothiophene',
    'c1ccc2c(c1)CNC2': '2,3-dihydro-1H-isoindole',
    'c1ccc2c(c1)COc1ccccc1-2': '6H-dibenzo[b,d]pyran',
    'c1ccc2c(c1)Cc1ccccc1O2': '9H-xanthene',
    'c1ccc2c(c1)Cc1ccccc1S2': '9H-thioxanthene',
    'c1ccc2c(c1)NCCN2': '1,2,3,4-tetrahydroquinoxaline',
    'c1ccc2c(c1)Nc1ccccc1O2': '10H-phenoxazine',
    'c1ccc2c(c1)Nc1ccccc1S2': '10H-phenothiazine',
    'c1ccc2c(c1)OCCO2': '2,3-dihydro-1,4-benzodioxine',
    'c1ccc2c(c1)Oc1ccccc1S2': 'phenoxathiine',  #: PIN keeps terminal 'e', the Blue Book)
    'c1ccc2c(c1)Sc1ccccc1S2': 'thianthrene',
    'c1ccc2c(c1)[nH]c1ccccc12': '9H-carbazole',
    'c1ccc2c(c1)[nH]c1cnccc12': '9H-beta-carboline',
    'c1ccc2c(c1)c1ccccc1c1ccccc21': 'triphenylene',
    'c1ccc2c(c1)ccc1[nH]ccc12': '3H-benzo[e]indole',
    'c1ccc2c(c1)ccc1c3ccccc3ccc21': 'chrysene',
    'c1ccc2c(c1)ccc1ncccc12': 'benzo[h]quinoline',
    'c1ccc2c(c1)ccc1occc12': 'naphtho[2,1-b]furan',
    'c1ccc2c(c1)ccc1sccc12': 'naphtho[2,1-b]thiophene',
    'c1ccc2c(c1)cnc1ccccc12': 'phenanthridine',
    'c1ccc2c(c1)oc1ccccc12': 'dibenzo[b,d]furan',
    # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value:
    # (the Blue Book) keeps a component's heteroatom locants in
    # square brackets, '[1]benzopyrano[2,3-c]pyrrole (PIN)' (:12157). OPSIN RT exact.
    'c1ccc2c(c1)oc1cccnc12': '[1]benzofuro[3,2-b]pyridine',
    'c1ccc2c(c1)sc1ccccc12': 'dibenzo[b,d]thiophene',
    'c1ccc2c[nH]cc2c1': '2H-isoindole',
    'c1ccc2cc3[nH]ccc3cc2c1': '1H-naphtho[2,3-b]pyrrole',
    # fix a performance pass (wp6-tests), change-asserted-value (was 'naphthacene'):
    # 'tetracene (PIN) (formerly naphthacene)' (the Blue Book). OPSIN RT exact.
    'c1ccc2cc3cc4ccccc4cc3cc2c1': 'tetracene',
    'c1ccc2cc3cnccc3cc2c1': 'benzo[g]isoquinoline',
    'c1ccc2cc3ncccc3cc2c1': 'benzo[g]quinoline',
    'c1ccc2cc3occc3cc2c1': 'naphtho[2,3-b]furan',
    'c1ccc2cc3sccc3cc2c1': 'naphtho[2,3-b]thiophene',
    'c1ccc2cnccc2c1': 'isoquinoline',
    'c1ccc2cnncc2c1': 'phthalazine',
    'c1ccc2cocc2c1': '2-benzofuran',  #: isobenzofuran->2-benzofuran PIN (BB 11829)
    'c1ccc2c(c1)COC2': '1,3-dihydro-2-benzofuran',  #: phthalan catalog add
    'c1ccc2nc3ccccc3cc2c1': 'acridine',
    'c1ccc2nc3ccccc3nc2c1': 'phenazine',
    'c1ccc2ncccc2c1': 'quinoline',
    'c1ccc2nccnc2c1': 'quinoxaline',
    'c1ccc2ncncc2c1': 'quinazoline',
    'c1ccc2nnccc2c1': 'cinnoline',
    'c1ccc2nocc2c1': '2,1-benzisoxazole',
    'c1ccc2nonc2c1': '2,1,3-benzoxadiazole',
    'c1ccc2nscc2c1': '2,1-benzothiazole',
    'c1ccc2nsnc2c1': '2,1,3-benzothiadiazole',
    'c1ccc2occc2c1': '1-benzofuran',  #: PIN locant, the Blue Book)
    'c1ccc2ocnc2c1': '1,3-benzoxazole',
    'c1ccc2oncc2c1': '1,2-benzisoxazole',
    'c1ccc2sccc2c1': '1-benzothiophene',  #: PIN locant, the Blue Book)
    'c1ccc2scnc2c1': '1,3-benzothiazole',
    'c1ccn2cccc2c1': 'indolizine',
    'c1ccn2ccnc2c1': 'imidazo[1,2-a]pyridine',
    'c1ccn2cncc2c1': 'imidazo[1,5-a]pyridine',
    'c1ccn2ncnc2c1': '[1,2,4]triazolo[1,5-a]pyridine',
    'c1ccn2nnnc2c1': '[1,2,3,4]tetrazolo[1,5-a]pyridine',
    'c1cn2ccsc2n1': 'imidazo[2,1-b]thiazole',
    'c1cnc2[nH]ccc2c1': '1H-pyrrolo[2,3-b]pyridine',
    'c1cnc2[nH]cnc2c1': '3H-imidazo[4,5-b]pyridine',
    'c1cnc2[nH]ncc2c1': '1H-pyrazolo[3,4-b]pyridine',
    'c1cnc2c(c1)CCN2': '2,3-dihydro-1H-pyrrolo[2,3-b]pyridine',
    'c1cnc2c(c1)ccc1cccnc12': '1,10-phenanthroline',
    'c1cnc2cc[nH]c2c1': '1H-pyrrolo[3,2-b]pyridine',
    'c1cnc2cccnc2c1': '1,5-naphthyridine',
    'c1cnc2ccncc2c1': '1,6-naphthyridine',
    'c1cnc2ccnn2c1': 'pyrazolo[1,5-a]pyrimidine',
    'c1cnc2ccnnc2c1': 'pyrido[3,2-c]pyridazine',
    'c1cnc2ccoc2c1': 'furo[3,2-b]pyridine',
    'c1cnc2ccsc2c1': 'thieno[3,2-b]pyridine',
    'c1cnc2cnccc2c1': '1,7-naphthyridine',
    'c1cnc2cnncc2c1': 'pyrido[2,3-d]pyridazine',
    'c1cnc2nccn2c1': 'imidazo[1,2-a]pyrimidine',
    'c1cnc2nccnc2c1': 'pyrido[2,3-b]pyrazine',
    'c1cnc2nccnc2n1': 'pyrazino[2,3-b]pyrazine',
    'c1cnc2ncncc2c1': 'pyrido[2,3-d]pyrimidine',
    'c1cnc2ncncc2n1': 'pteridine',
    'c1cnc2ncoc2c1': 'oxazolo[4,5-b]pyridine',
    'c1cnc2ncsc2c1': 'thiazolo[4,5-b]pyridine',
    'c1cnc2nncn2c1': '[1,2,4]triazolo[4,3-a]pyrimidine',
    'c1cnc2nocc2c1': 'isoxazolo[3,4-b]pyridine',
    'c1cnc2occc2c1': 'furo[2,3-b]pyridine',
    'c1cnc2ocnc2c1': 'oxazolo[5,4-b]pyridine',
    'c1cnc2oncc2c1': 'isoxazolo[5,4-b]pyridine',
    'c1cnc2sccc2c1': 'thieno[2,3-b]pyridine',
    'c1cnc2scnc2c1': 'thiazolo[5,4-b]pyridine',
    'c1cnc2scnc2n1': 'thiazolo[4,5-b]pyrazine',
    'c1cnn2cccc2c1': 'pyrrolo[1,2-b]pyridazine',
    'c1cnn2ccnc2c1': 'imidazo[1,2-b]pyridazine',
    'c1ncc2[nH]ccc2n1': '5H-pyrrolo[3,2-d]pyrimidine',
    'c1ncc2[nH]cnc2n1': '1H-imidazo[4,5-d]pyrimidine',
    'c1ncc2nc[nH]c2n1': '9H-purine',
    'c1ncc2ncncc2n1': 'pyrimido[5,4-d]pyrimidine',
}


_PIN_NOT_BUILT = {
    'C1CCN2CCCCC2C1': "needs the hydro + indicated-hydrogen quinolizine parent "
                      "(octahydro-2H-quinolizine); the catalog name 'quinolizidine' is "
                      "not a Blue Book name and ships labelled below pin_verified",
    'C1CC2CCCN2C1': "needs the hydro + indicated-hydrogen pyrrolizine parent "
                    "(hexahydro-1H-pyrrolizine); the catalog name 'pyrrolizidine' is not a "
                    "Blue Book name and ships labelled below pin_verified",
    'C1CCN2CCCC2C1': "needs the hydro indolizine parent (octahydroindolizine); the "
                     "catalog name 'indolizidine' is not a Blue Book name and ships "
                     "labelled below pin_verified",
    'C1=Cn2cccc2C1': "DEFECT (non-PIN at pin_verified): the catalog spells the mancude "
                     "system 'pyrrolizine' without its indicated hydrogen; BB Table 2.8 "
                     "(21) 'the PIN is 1H-pyrrolizine' (:11628). .planning/"
                     "TODO-2026-09-24.md 'Open from T12 fix round 2 (wp6)'",
}


@pytest.mark.parametrize(
    "smiles,expected_name",
    [pytest.param(k, v, marks=pytest.mark.xfail(strict=True, reason=_PIN_NOT_BUILT[k]))
     if k in _PIN_NOT_BUILT else (k, v)
     for k, v in POST_148_2_BASELINE_NAMES.items()],
    ids=list(POST_148_2_BASELINE_NAMES.keys()),
)
def test_catalog_byte_identical(smiles, expected_name):
    """: cataloged-path output byte-identical post-149.

    Source: 149-internal notes.
    Source: / G6 acceptance gate.
    """
    actual = name_compound(smiles)
    assert actual == expected_name, (
        f"Phase 149 D-11 byte-identical lock VIOLATED for {smiles!r}: "
        f"expected {expected_name!r}, got {actual!r}"
    )


def test_baseline_table_size():
    """Sanity check: every frozen baseline key is still a catalog entry.

     fix a performance pass (wp6-tests), TEST-BUG: this asserted len(baseline) ==
    len(FUSED_HETEROCYCLE_DATA), which could only hold at capture time (157 then; the
    catalog has grown to 219 entries since, with no baseline key missing). The frozen
    table is a snapshot of the entries that existed then; what it can check is that
    none of them has left the catalog. The newer entries are covered by the catalog
    round-trip tests (tests/unit/data/test_fused_het_data_integrity.py)."""
    missing = sorted(set(POST_148_2_BASELINE_NAMES) - set(FUSED_HETEROCYCLE_DATA))
    assert not missing, f"frozen baseline keys no longer in FUSED_HETEROCYCLE_DATA: {missing}"
