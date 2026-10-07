"""Lane L2 proper fix: an acyl prefix carries its writer's facts -- multiplier and marks --
decided from the acyl group's atoms; no vocabulary or character test decides them.

* (the Blue Book) marks around compound and complex prefixes:
  '(chloroacetyl)oxyl (PIN)' (:40694), '2-(acetyloxy)ethane-1-sulfonic acid (PIN)'
  (:31713), '2-[(methylcarbamoyl)amino]...' (:33354);
* the suffix 'carbonyl' on a ring,:30624) is enclosed but simple:
  '4-(cyclohexanecarbonyl)benzene-1-carbothioic acid (PIN)' (:7322),
  'N,N-di(cyclohexanecarbonyl)cyclohexanecarboxamide (PIN)' (:33115);
* the retained 'benzoyl' and the alkanoyls are bare: '3-(benzoyloxy)propanoic acid (PIN)'
  (:31711); 'acetyl (preferred prefix)' (:30442).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import name_is_rt_exact

# acyl on a benzene ring: the parent atom is 5, the carbonyl carbon is 6
ACYLS = [
    ("c1ccccc1C=O", False, False),                  # formyl
    ("c1ccccc1C(C)=O", False, False),               # acetyl
    ("c1ccccc1C(=O)CC", False, False),              # propanoyl
    ("c1ccccc1C(=O)/C=C/C", False, None),           # but-2-enoyl: marks left to main
    ("c1ccccc1C(=O)CCl", True, True),               # chloroacetyl
    ("c1ccccc1C(=O)c1ccccc1", False, False),        # benzoyl (retained)
    ("c1ccccc1C(=O)c1ccc(Cl)cc1", True, True),      # 4-chlorobenzoyl
    ("c1ccccc1C(=O)C1CCCCC1", False, True),         # cyclohexanecarbonyl
    ("c1ccccc1C(=O)c1cccnc1", False, True),         # pyridine-3-carbonyl
    ("c1ccccc1C(=O)N1CCC1", False, True),           # azetidine-1-carbonyl
    ("c1ccccc1C(=O)c1ccc(-c2ccccc2)cc1", False, True),   # [1,1'-biphenyl]-4-carbonyl
]


@pytest.mark.parametrize("smiles,substituted,enclosed", ACYLS)
def test_acyl_derivation_from_the_atoms(smiles, substituted, enclosed):
    from orthonym.assembly.book_prefixes import acyl_derivation
    from orthonym.assembly.prefix_derivation import PrefixDerivation
    mol = Chem.MolFromSmiles(smiles)
    acyl = {a.GetIdx() for a in mol.GetAtoms()} - {0, 1, 2, 3, 4, 5}
    assert acyl_derivation(mol, 6, acyl) == PrefixDerivation(substituted=substituted,
                                                             enclosed=enclosed)


@pytest.mark.parametrize("text,substituted,enclosed,expected", [
    ("cyclohexanecarbonyl", False, True, "(cyclohexanecarbonyl)"),
    ("acetyl", False, False, "acetyl"),
    ("chloroacetyl", True, True, "(chloroacetyl)"),
    ("acetyloxy", True, True, "(acetyloxy)"),
])
def test_the_enclosure_predicate_reads_the_record(text, substituted, enclosed, expected):
    from orthonym.assembly.naming_utils import enclose_if_compound
    from orthonym.assembly.prefix_derivation import built
    assert enclose_if_compound(built(text, substituted=substituted, enclosed=enclosed)) \
        == expected


def test_fg_enclose_reads_the_record():
    from orthonym.assembly.prefix_derivation import built
    from orthonym.assembly.substituent_naming import _fg_enclose
    assert _fg_enclose(built("methylcarbamoyl", substituted=True, enclosed=True)) \
        == "(methylcarbamoyl)"
    assert _fg_enclose(built("carbamoyl", substituted=False, enclosed=False)) == "carbamoyl"


def test_the_urea_composer_reads_the_record():
    # HEAD: 'methylcarbamoylurea' (the character test saw no mark);:7232
    from orthonym.assembly.composer import _build_n_substituted_name
    from orthonym.assembly.prefix_derivation import built
    name = built("methylcarbamoyl", substituted=True, enclosed=True)
    assert _build_n_substituted_name([("N", name)], "urea") == "(methylcarbamoyl)urea"


def test_no_vocabulary_pattern_decides_an_acyl_prefix():
    from orthonym.assembly import naming_utils as nu
    assert not hasattr(nu, "_BARE_ACYL_PREFIX_RE")


def _row(smiles, tier):
    eng = Orthonym() if tier == "pin" else Orthonym(style="pin", **_emit_tier_flags(tier))
    return eng.name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,name", [
    # the urea positives of the deleted character test (study-audit H; names on HEAD)
    ("NC(=O)NC(O)c1ccccc1", "pin", "[hydroxy(phenyl)methyl]urea"),
    ("CN(C)C(=O)NC(O)c1ccccc1", "pin", "N'-[hydroxy(phenyl)methyl]-N,N-dimethylurea"),
    ("O=C(NC(O)c1ccccc1)Nc1ccccc1", "pin", "N-[hydroxy(phenyl)methyl]-N'-phenylurea"),
    ("NC(=O)NCNC(N)=O", "pin", "N-[(carbamoylamino)methyl]urea"),
    # acyl-oxy and acyl-amino positives of the deleted vocabulary test (study-audit G)
    ("OC(=O)CCCCOC(=O)CCl", "best-effort", "5-[(chloroacetyl)oxy]pentanoic acid"),
    ("CC(=O)NCCSC(=O)/C=C/CCC[C@H](O)/C=C/CC(C)O", "best-effort",
     "S-[2-(acetylamino)ethyl] (2E,7S,8E)-7,11-dihydroxydodeca-2,8-dienethioate"),
])
def test_engine_names_keep_their_marks(smiles, tier, name):
    row = _row(smiles, tier)
    assert row.get("name") == name, row
    assert name_is_rt_exact(name, smiles)
