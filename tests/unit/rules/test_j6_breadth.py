"""Suite fix j6-breadth (TRIAGE.md 'Suite fix -- j6-breadth'): names the job's
producer fixes add where the PIN tier abstained, each asserted exactly (gate on,
PIN tier) with an independent full-InChIKey OPSIN round trip.

Blue Book basis per group:
- ring-attached imidic acid on benzene: (the Blue Book,
  'benzenecarboximidic acid (PIN)':30000); with other substituents the group
  keeps '-1-': '2-(propanimidoylselanyl)benzene-1-carboximidic acid (PIN)'
  (:32013); demoted: '4-(C-hydroxycarbonimidoyl)benzoic acid (PIN)' (:30033).
- branched unsaturated substituent chains: (:15800), (b)
  (:20916), (:21033); parent by (:18875).
- ring branch on a chain substituent: (:15800), (:3477),
   (:7444); one-atom terminal hetero prefixes: Appendix 2 (:56167,
  :55458,:56799).
- 2-component fusion names of an N-H system whose bare name OPSIN reads with a
  different indicated hydrogen: descriptor (:11911); indicated
  hydrogen cited, (:14607). The standard InChIKey does not place
  a mobile hydrogen, so these rows also compare canonical SMILES.
"""
import pytest

from orthonym.namer import Orthonym
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

PIN_NAMES = [
    # g8 C23: benzene carboximidic acid suffix
    ("N=C(O)c1ccccc1", "benzenecarboximidic acid"),
    ("N=C(O)c1ccc(C)cc1", "4-methylbenzene-1-carboximidic acid"),
    ("N=C(O)c1ccccc1Cl", "2-chlorobenzene-1-carboximidic acid"),
    ("N=C(O)c1ccc(O)cc1", "4-hydroxybenzene-1-carboximidic acid"),
    ("N=C(O)c1ccc(S(=O)(=O)O)cc1", "4-sulfobenzene-1-carboximidic acid"),
    ("N=C(O)c1ccc(C(N)=O)cc1", "4-carbamoylbenzene-1-carboximidic acid"),
    ("N=C(O)c1ccc(C(=N)O)cc1", "benzene-1,4-dicarboximidic acid"),
    ("N=C(O)c1ccc(C(=O)O)cc1", "4-(C-hydroxycarbonimidoyl)benzoic acid"),
]

HYDROXYALKENYL_NAMES = [
    # g2 G2-C8: branched unsaturated substituent chain, numbered from the free
    # valence; longest chain first, then more multiple bonds
    # (b):20916,:21033); the ring with more -OH is the parent
    #:18875).
    ("CC(C)(O)/C=C/c1cc(O)ccc1O",
     "2-[(1E)-3-hydroxy-3-methylbut-1-en-1-yl]benzene-1,4-diol"),
    ("CC(C)(O)/C=C/c1ccc(O)cc1", "4-[(1E)-3-hydroxy-3-methylbut-1-en-1-yl]phenol"),
    ("CC(C)(O)C=Cc1ccc(O)cc1", "4-(3-hydroxy-3-methylbut-1-en-1-yl)phenol"),
    ("CC(C)(Cl)/C=C/c1ccc(O)cc1", "4-[(1E)-3-chloro-3-methylbut-1-en-1-yl]phenol"),
    ("CC(C)(O)/C=C/C=C/c1ccc(O)cc1",
     "4-[(1E,3E)-5-hydroxy-5-methylhexa-1,3-dien-1-yl]phenol"),
    ("C=C(C)C(O)c1ccc(O)cc1", "4-(1-hydroxy-2-methylprop-2-en-1-yl)phenol"),
    # the chain stereocentre keeps its locant,:44643)
    ("C=C[C@](C)(O)CCc1ccc(O)cc1", "4-[(3R)-3-hydroxy-3-methylpent-4-en-1-yl]phenol"),
]

RING_BRANCH_NAMES = [
    # g4 C5: a ring branch on a chain substituent is a prefix of the chain
    # substituent, which keeps the free valence; order.
    ("C/C(=C\\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO",
     "(2Z,5E,9E)-2-[2-(2,5-dihydroxyphenyl)-2-oxoethyl]-11-hydroxy-6,10-"
     "dimethylundeca-2,5,9-trienoic acid"),
    ("OC(=O)c1ccc(CC(=O)c2ccccc2)cc1", "4-(2-oxo-2-phenylethyl)benzoic acid"),
    ("OC(=O)CCC(CC(=O)c1ccccc1)C(=O)O", "2-(2-oxo-2-phenylethyl)pentanedioic acid"),
    ("OC(=O)C(CC(O)c1ccccc1)CCC", "2-(2-hydroxy-2-phenylethyl)pentanoic acid"),
    ("OC(=O)c1ccc(C(O)c2ccccc2)cc1", "4-[hydroxy(phenyl)methyl]benzoic acid"),
    ("OC(=O)c1ccc(CC(=O)c2ccc(Cl)cc2)cc1",
     "4-[2-(4-chlorophenyl)-2-oxoethyl]benzoic acid"),
    ("OC(=O)CCC(CC(O)c1ccccn1)C(=O)O",
     "2-[2-hydroxy-2-(pyridin-2-yl)ethyl]pentanedioic acid"),
    # g5 C11 PAH_08: a one-atom terminal hetero substituent on the complex-ring
    # path is its preferred prefix (Appendix 2,:56167 hydroxy,:55458 amino,
    #:56799 sulfanyl)
    ("CC(C)[C@H]1CC[C@@H](CO)c2c(O)cc(C(=O)O)cc21",
     "(5R,8R)-4-hydroxy-5-(hydroxymethyl)-8-(propan-2-yl)-5,6,7,8-"
     "tetrahydronaphthalene-2-carboxylic acid"),
    ("Oc1cc(C(=O)O)cc2c1CCCC2", "4-hydroxy-5,6,7,8-tetrahydronaphthalene-2-carboxylic acid"),
    ("Nc1cc(C(=O)O)cc2c1CCCC2", "4-amino-5,6,7,8-tetrahydronaphthalene-2-carboxylic acid"),
    ("Sc1cc(C(=O)O)cc2c1CCCC2",
     "4-sulfanyl-5,6,7,8-tetrahydronaphthalene-2-carboxylic acid"),
]

INTERNAL_AND_MARKS_NAMES = [
    # g5 C11 Aminobenzoic_35: internal free valence on a linear FG chain, free
    # valence lowest; inner mark escalation:7444)
    ("CC(=O)[C@@H](C)Nc1ccccc1C(=O)O", "2-{[(2R)-3-oxobutan-2-yl]amino}benzoic acid"),
    ("CC(=O)C(C)Nc1ccccc1C(=O)O", "2-[(3-oxobutan-2-yl)amino]benzoic acid"),
    ("OC(=O)c1ccc(C(C)C(C)=O)cc1", "4-(3-oxobutan-2-yl)benzoic acid"),
    ("OC(=O)c1ccc(C(C(C)=O)C(C)=O)cc1", "4-(2,4-dioxopentan-3-yl)benzoic acid"),
    ("OC(=O)c1ccc(C(CCl)CC(=O)C)cc1", "4-(1-chloro-4-oxopentan-2-yl)benzoic acid"),
    ("C[C@@H](Cl)Nc1ccc(C(=O)O)cc1", "4-{[(1R)-1-chloroethyl]amino}benzoic acid"),
    # the indicated hydrogen sits at the ketone, hydro prefixes at the other
    # saturated positions: (the Blue Book); was '...-2,3-
    # dihydro-1H-indol-2-one' (TRIAGE j12 finding 5)
    ("CC1=CC=C(C=C1)CN2C3=CC=CC=C3[C@@](C2=O)(CC(=O)C4=C(C=C(C=C4)C)C)O",
     "(3R)-3-[2-(2,4-dimethylphenyl)-2-oxoethyl]-3-hydroxy-1-[(4-methylphenyl)"
     "methyl]-1,3-dihydro-2H-indol-2-one"),
    # one-carbon substituent: descriptor without locant:45031,
    #:44643)
    ("OC(=O)c1ccc([C@H](O)Cl)cc1", "4-[(R)-chloro(hydroxy)methyl]benzoic acid"),
    ("O=c1ccoc([C@H](O)c2ccccc2)c1", "2-[(R)-hydroxy(phenyl)methyl]-4H-pyran-4-one"),
    ("OC(=O)c1ccc([C@@H](O)c2ccccc2)cc1", "4-[(S)-hydroxy(phenyl)methyl]benzoic acid"),
]

SATURATED_CHALCOGEN_RING_NAMES = [
    # g3 C17b rt75_170: decorated saturated chalcogen heteromonocycles take the
    # Hantzsch-Widman stem:8224), heteroatoms lowest then the
    # free valence
    ("C=C[C@](C)(O)CCC=C(C)CCC1OC(C)(C)OC1(C)C",
     "(3R)-3,7-dimethyl-9-(2,2,5,5-tetramethyl-1,3-dioxolan-4-yl)nona-1,6-dien-3-ol"),
    ("CC1(C)OCC(CO)O1", "(2,2-dimethyl-1,3-dioxolan-4-yl)methanol"),
    ("OC(=O)CCC1COC(C)O1", "3-(2-methyl-1,3-dioxolan-4-yl)propanoic acid"),
    ("OC(=O)CCC1OCC(C)(C)CO1", "3-(5,5-dimethyl-1,3-dioxan-2-yl)propanoic acid"),
    ("OC(=O)CCC1COC(C)(C)OC1", "3-(2,2-dimethyl-1,3-dioxan-5-yl)propanoic acid"),
    ("OC(=O)CCC1CSC(C)S1", "3-(2-methyl-1,3-dithiolan-4-yl)propanoic acid"),
    ("OC(=O)CCC1COCC(C)O1", "3-(6-methyl-1,4-dioxan-2-yl)propanoic acid"),
    ("OC(=O)CCC1CCC(C)S1", "3-(5-methylthiolan-2-yl)propanoic acid"),
]

FUSION_NAMES = [
    # g3 C08: pyrrole c-fused to an azine, N-H cited as indicated hydrogen
    ("c1ncc2c[nH]cc2n1", "6H-pyrrolo[3,4-d]pyrimidine"),
    ("c1cc2c[nH]cc2nn1", "6H-pyrrolo[3,4-c]pyridazine"),
    ("c1cnc2c[nH]cc2c1", "6H-pyrrolo[3,4-b]pyridine"),
    ("c1cc2c[nH]cc2cn1", "2H-pyrrolo[3,4-c]pyridine"),
    ("Cc1ncc2c[nH]cc2n1", "2-methyl-6H-pyrrolo[3,4-d]pyrimidine"),
    ("Cn1cc2cncnc2c1", "6-methyl-6H-pyrrolo[3,4-d]pyrimidine"),
]


@pytest.mark.parametrize("smiles,expected",
                         PIN_NAMES + HYDROXYALKENYL_NAMES + RING_BRANCH_NAMES
                         + INTERNAL_AND_MARKS_NAMES
                         + SATURATED_CHALCOGEN_RING_NAMES)
def test_pin_name(smiles, expected):
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles), expected


def test_imidic_acid_never_outranks_a_thioic_s_acid():
    """A carbothioic S-acid is senior to an imidic acid; SENIORITY_ORDER
    rank 2 vs 11) though the benzene suffix list does not rank it: the thioic
    acid stays the suffix and the imidic acid is the 'C-hydroxycarbonimidoyl'
    prefix, as before the imidic acid had a suffix form on benzene."""
    name = Orthonym().name("OC(=N)c1ccc(C(=O)S)cc1")
    assert "carboximidic acid" not in name, name
    assert "C-hydroxycarbonimidoyl" in name and "carbothioic S-acid" in name, name


@pytest.mark.parametrize("smiles,expected", FUSION_NAMES)
def test_fusion_pin_name(smiles, expected):
    from rdkit import Chem
    from tests.support.rt_assert import _independent_parse
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles), expected
    parsed = _independent_parse(expected)
    assert parsed and (Chem.MolToSmiles(Chem.MolFromSmiles(parsed))
                       == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))), parsed


def test_dihydro_pyrrolopyrrole_is_not_named_bare():
    """Both N-H of pyrrolo[3,4-b]pyrrole: a hydro input (the mancude parent has
    no indicated hydrogen), so the skeleton-matched descriptor must not ship
    as the bare 'pyrrolo[3,4-b]pyrrole' (a different molecule)."""
    from rdkit import Chem
    from orthonym.rules.fused_rings import _try_algorithmic_fusion_name
    assert _try_algorithmic_fusion_name(Chem.MolFromSmiles("c1cc2c[nH]cc2[nH]1")) is None


@pytest.mark.parametrize("smi", [
    # '...[methyl(3-methyl-1,2-dioxobutyl)amino]...' (PIN acyl: 3-methyl-2-oxobutanoyl)
    "CC(C)C(=O)C(=O)N(C)[C@H](C(=O)NCCc1c[nH]c2ccccc12)C(C)C",
    # '2,5-bis[(aminoamino)oxomethyl]bicyclo[2.2.1]heptane' (the carbohydrazide)
    "C1[C@@H]2C[C@@H]([C@H]1C[C@@H]2C(=O)NN)C(=O)NN",
    # '...-1-{[(3-methylbutyl)amino]oxomethyl}-...' (also a suffix-less
    # partially saturated name)
    "CCC[C@]1(O)C=C[C@@H]2CC=CC[C@@H]2[C@H]1C(=O)NCCC(C)C",
])
def test_oxo_on_free_valence_acyl_form_is_not_labelled_pin(smi):
    """A '1-oxo...yl' / '...oxomethyl' acyl spelling (oxo on the free-valence
    carbon with another group) is not the PIN -- the acyl/carbamoyl prefix is
    , -- so the name ships below pin_verified (honest
    demotion; these two m1500 rows were pin_verified before j6) and still
    round-trips."""
    r = Orthonym().name_tiered(smi)
    if r["tier"] == "abstain":
        return
    assert r["tier"] != "pin_verified", r
    assert name_is_rt_exact(r["name"], smi), r


@pytest.mark.parametrize("smiles", [
    "ClC(Cl)(c1ccccc1)c1ccccc1",                   # Ph-CCl2-Ph: multiplicative
    "OC(=O)c1ccc(CC(=O)c2ccc(C(=O)O)cc2)cc1",      # both rings carry the suffix
    "ClC(Cc1ccccc1)c1ccccc1",                      # Ph-CH2-CHCl-Ph
    "ClC(Cl)(Cc1ccccc1)c1ccccc1",
    "BrCC(Cl)(c1ccccc1)c1ccccc1",
])
def test_ring_branch_does_not_preempt_multiplicative(smiles):
    """A ring branch that would make the molecule a multiplicative candidate
    (identical units, is not named by the ring-branch producer: the
    PIN tier must not ship a substitutive '(...-phenylethyl)benzene' /
    '...(phenyl)methyl]benzene' / '4-[2-(4-carboxyphenyl)-2-oxoethyl]benzoic
    acid' at pin_verified (without the guard: '(1-chloro-2-phenylethyl)
    benzene' shipped pin_verified)."""
    r = Orthonym().name_tiered(smiles)
    assert r["tier"] != "pin_verified" or not (
        "phenyl)methyl" in r["name"] or "carboxyphenyl" in r["name"]
        or "phenylethyl)benzene" in r["name"]), r


def test_substituted_ring_chalcogen_is_not_a_plain_stem():
    """A ring S carrying a methyl is lambda-4: the Hantzsch-Widman
    stem path does not cover it."""
    r = Orthonym().name_tiered("OC(=O)CCC1COS(C)O1")
    assert "dioxathiolan" not in (r["name"] or "") or "λ" in r["name"], r


# The rows of the j6 causes whose PIN the PIN tier cannot build yet (each raw
# check is a strict xfail in its own file, naming the missing producer). What
# ships, gate on: best-effort names the molecule RT-exact; the PIN tier fails
# closed or ships an RT-exact name (tests/support/rt_assert.assert_tier_contract).
PIN_NOT_BUILT_ROWS = {
    "macrolide_16_ring": "C/C1=C/C[C@@H](/C(C)=C/c2csc(C)n2)OC(=O)C[C@H](O)"
                         "C(C)(C)C(=O)[C@H](C)[C@@H](O)/C(C)=C/CC1",
    "indolone_piperazinedione": "C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)"
                                "c3ccccc32)NC1=O",
    "trioxa_spiro_terpene": "C=C1CC[C@@H](C/C=C2/CC[C@]3(OC2)O[C@@]2(O)CC[C@]3(C)"
                            "OC2(C)C)C(C)(C)[C@H]1[C@@H](O)C=C1CCOC1=O",
    "decalin_ketone": "CC(=O)[C@@]1(C)C(C)=C[C@H](O)[C@H]2C[C@](C)(O)CC[C@@H]21",
    "stereo_free_stigmastane": "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C",
    "stereo_free_ergostane": "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C",
    "cholestane_sulfate": "CC(CCCC(C)(O)COS(=O)(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)"
                          "C[C@@H]4C[C@H](O)CCC4(C)[C@H]3C[C@H](O)C12C",
    "benzo_fused_aza_acid": "O=C(O)c1cc2cc3c4c(c2oc1=O)CCCN4CCC3",
    "anthracenedione_methoxy": "COc1cccc2c1C(=O)c1ccc3c(c1C2=O)C(=O)C[C@@H](C)[C@H]3O",
    "anthraquinone_polyhydroxy": "COc1cc(O)c2c(c1O)C(=O)c1c(C(C)=O)c(O)cc(O)c1C2=O",
    "dihydrofuran_dione_diacid": "COC(=O)/C(CC(=O)O)=C(\\CCCCCCCCCCCCCCCCC1=C(C)"
                                 "C(=O)OC1=O)C(=O)O",
    "tetracyclic_diaza": "O=C1N[C@@H](C[C@@]2(O)c3ccccc3N3C(=O)[C@@H]4CCCCN4"
                         "[C@@H]32)C(=O)N[C@H]1Cc1ccccc1",
    "stereo_free_androstane": "CC12CCCC1C1CCC3CCCCC3(C)C1CC2",
    "stereo_free_cholesterol": "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C",
    "triquinane": "COCC1=C2[C@@H]3CC(C)(C)C[C@@H]3C[C@@]2(O)CC1=O",
    "diglucosyl_chromane": "OCC1OC(Oc2cc(O)c3c(c2)OC(c2ccc(O)c(OC4OC(CO)C(O)C(O)"
                           "C4O)c2)C(O)C3)C(O)C(O)C1O",
    "cyclohexylidene_acid": "CC1C/C(=C\\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1",
    "benzothiophene_amine_hcl": "Cl.c1ccc2sc(C3(N4CCCCC4)CCCCC3)cc2c1",
    "hexahydronaphthalene": "C=C(C)C1C=C2C(C)=CCCC2(C)CC1",
    "furanone_tricycle": "COC1C2=C(C)C(=O)OC2CC2CCC(O)C(C)C21C",
    "azulene_like_tricycle": "COC[C@@]1(O)CC[C@@H]2C1=C[C@]1(C)C(=C(C(C)C)C[C@H]1O)"
                             "C[C@H](O)[C@@H]2C",
    "stereo_free_androstanolone": "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O",
    "benzoxacyclododecinone": "COc1cc(O)cc2c1C(=O)O[C@@H](C)CCCCC/C=C/2",
    "flavone_di_c_glycoside": "CC1OC(Oc2c(C3OC(CO)C(O)C(O)C3O)c(O)c3c(=O)cc(-c4ccc(O)"
                              "c(O)c4)oc3c2C2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O",
    "bridged_depsipeptide": "CCCCCC(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H]1C(=O)N[C@@H]"
                            "(CCCN=C(N)N)C(=O)N[C@H]2CC[C@@H](O)N(C2=O)[C@@H]"
                            "([C@@H](C)CC)C(=O)N(C)[C@@H](CC(=O)c2ccccc2N)C(=O)"
                            "N[C@@H](C(C)C)C(=O)O[C@@H]1C",
}


@pytest.mark.parametrize("smiles", list(PIN_NOT_BUILT_ROWS.values()),
                         ids=list(PIN_NOT_BUILT_ROWS))
def test_pin_not_built_rows_keep_the_tier_contract(smiles):
    from tests.support.rt_assert import assert_tier_contract
    assert_tier_contract(smiles)


@pytest.mark.parametrize("smiles", [
    "C/C(=C\\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO",
    "OC(=O)c1ccc(CC(=O)c2ccc(Cl)cc2)cc1",
    "CC(C)(O)/C=C/c1cc(O)ccc1O",
    "CC(=O)[C@@H](C)Nc1ccccc1C(=O)O",
    "C=C[C@](C)(O)CCC=C(C)CCC1OC(C)(C)OC1(C)C",
    "c1ncc2c[nH]cc2n1",
])
def test_new_producers_do_not_depend_on_atom_order(smiles):
    """Names never depend on input atom order: the canonical spelling and eight
    seeded random spellings give one name (the ring-branch host scan once
    declined when a ring's own -OH came before its host carbon)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    spellings = {Chem.MolToSmiles(mol)}
    for seed in range(8):
        spellings.add(Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, _perm(mol.GetNumAtoms(), seed)),
            canonical=False))
    names = {Orthonym().name(s) for s in spellings}
    assert len(names) == 1, names


def _perm(n, seed):
    import random
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


@pytest.mark.parametrize("smiles", [
    "NC1CCc2ccc(O)cc2C1",        # PIN '7-amino-5,6,7,8-tetrahydronaphthalen-2-ol'
    "NC(=O)C1CCc2cc(O)ccc2C1",   # PIN '6-hydroxy-...-2-carboxamide'
    "NC(=O)C1CCc2ccccc2C1",      # PIN '1,2,3,4-tetrahydronaphthalene-2-carboxamide'
    "N#CC1CCc2ccccc2C1",         # PIN '...-2-carbonitrile'
])
def test_no_suffix_less_name_at_pin_verified(smiles):
    """A suffix-capable principal characteristic group is never cited only as a
    prefix at pin_verified,: Step 0a declines a one-atom -OH /
    -NH2 of the principal class, and the partially saturated carbocycle path
    records its suffix-less names non-PIN. Whatever ships round-trips."""
    r = Orthonym().name_tiered(smiles)
    if r["tier"] == "abstain":
        return
    assert r["tier"] != "pin_verified", r
    assert name_is_rt_exact(r["name"], smiles), r
