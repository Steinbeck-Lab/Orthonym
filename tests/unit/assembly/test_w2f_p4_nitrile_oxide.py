"""/2 neutral nitrile-oxide functional-class suffix (W2F p4).

BB (the Blue Book): R-C#NO 'nitrile oxides' are named by method
(1) — the word 'oxide' appended to the nitrile name — which yields the PIN.
Examples: 'benzonitrile oxide' (34876), 'acetonitrile oxide' (43285). Nitrile
oxides are classed with zwitterions, so they are senior to esters/acids.

: the BB ester PIN '4-(methoxycarbonyl)benzonitrile oxide' (34893)
is now BUILT — the aromatic-benzene forced-nitrile fix (_assemble_ring_nitrile_name
delegates aromatic benzene rings to the benzonitrile assembler) plus the
'(methoxycarbonyl)' enclosing marks landed. The acid variant still
fails closed (the forced-nitrile override does not demote carboxylic_acid).

Under pytest the OPSIN validity gate is disabled (conftest autouse); the positive
cases below emit the correct name regardless (verified RT-clean with the gate ON),
and the fail-closed cases are asserted at the handler predicate/return level.
"""
from rdkit import Chem

import orthonym
from orthonym.assembly.handlers.nitrile_oxide import (
    _is_nitrile_oxide, name_nitrile_oxide,
)


class _Feat:
    """Minimal features stub carrying only.mol (all the handler's decline paths
    read on the ester/acid guards need)."""
    def __init__(self, mol):
        self.mol = mol


class TestNitrileOxide:
    def test_benzonitrile_oxide(self):
        assert orthonym.name_compound("c1ccccc1C#[N+][O-]", style="pin") == "benzonitrile oxide"

    def test_acetonitrile_oxide(self):
        # aliphatic PIN (the Blue Book)
        assert orthonym.name_compound("CC#[N+][O-]", style="pin") == "acetonitrile oxide"

    def test_chloro_variant(self):
        assert orthonym.name_compound("Clc1ccc(C#[N+][O-])cc1", style="pin") == "4-chlorobenzonitrile oxide"

    def test_hydroxy_variant(self):
        assert orthonym.name_compound("Oc1ccc(C#[N+][O-])cc1", style="pin") == "4-hydroxybenzonitrile oxide"

    def test_amino_variant(self):
        assert orthonym.name_compound("Nc1ccc(C#[N+][O-])cc1", style="pin") == "4-aminobenzonitrile oxide"

    def test_acetyl_variant(self):
        assert orthonym.name_compound("CC(=O)c1ccc(C#[N+][O-])cc1", style="pin") == "4-acetylbenzonitrile oxide"

    def test_aliphatic_ring(self):
        # genuine aliphatic cyclohexane ring — the cyclohex guard must NOT fire
        assert orthonym.name_compound("C1CCCCC1C#[N+][O-]", style="pin") == "cyclohexanecarbonitrile oxide"

    def test_fused_arene(self):
        assert orthonym.name_compound("c1ccc2cc(C#[N+][O-])ccc2c1", style="pin") == "naphthalene-2-carbonitrile oxide"

    def test_ester_names_correctly(self):
        #: the Blue Book verbatim PIN. The senior nitrile oxide demotes
        # the ester to the '(methoxycarbonyl)' prefix, enclosed;
        # the aromatic benzene ring is named as a benzonitrile, not cyclohexane.
        mol = Chem.MolFromSmiles("COC(=O)C1=CC=C(C#[N+][O-])C=C1")
        assert _is_nitrile_oxide(_Feat(mol)) is True
        assert (orthonym.name_compound("COC(=O)C1=CC=C(C#[N+][O-])C=C1", style="pin")
                == "4-(methoxycarbonyl)benzonitrile oxide")

    def test_acid_fails_closed(self):
        # nitrile oxide is senior to the acid, but the forced-nitrile override does
        # not demote carboxylic_acid (-> '4-cyanobenzoic acid'); the ends-with-
        # 'nitrile' guard declines -> fail closed (buildable follow-on).
        mol = Chem.MolFromSmiles("OC(=O)c1ccc(C#[N+][O-])cc1")
        feat = _Feat(mol)
        assert _is_nitrile_oxide(feat) is True
        assert name_nitrile_oxide(feat, mol=mol) is None

    def test_predicate_neutral_only(self):
        # the deprotonated carboxylate salt is net-charge < 0 -> predicate declines
        # (the anion prefix path owns it), and it is a multi-fragment salt too.
        mol = Chem.MolFromSmiles("[Na+].[O-]C(=O)c1ccc(C#[N+][O-])cc1")
        assert _is_nitrile_oxide(_Feat(mol)) is False

    def test_predicate_declines_plain_nitrile(self):
        # a plain nitrile (no [O-]) is not a nitrile oxide
        mol = Chem.MolFromSmiles("c1ccccc1C#N")
        assert _is_nitrile_oxide(_Feat(mol)) is False


class TestNitrileChalcogenides:
    """: the heavier-chalcogen analogues R-C#[N+]-[X-] (X = S/Se/Te)
    are named the same way as the nitrile oxide, the chalcogen word replacing
    'oxide' (verified RT-clean with the OPSIN gate ON)."""

    def test_acetonitrile_sulfide(self):
        assert orthonym.name_compound("CC#[N+][S-]", style="pin") == "acetonitrile sulfide"

    def test_acetonitrile_selenide(self):
        assert orthonym.name_compound("CC#[N+][Se-]", style="pin") == "acetonitrile selenide"

    def test_acetonitrile_telluride(self):
        assert orthonym.name_compound("CC#[N+][Te-]", style="pin") == "acetonitrile telluride"

    def test_benzonitrile_sulfide(self):
        assert orthonym.name_compound("c1ccccc1C#[N+][S-]", style="pin") == "benzonitrile sulfide"

    def test_oxide_unchanged(self):
        # the O sibling is byte-identical to the pre-generalisation output
        assert orthonym.name_compound("CC#[N+][O-]", style="pin") == "acetonitrile oxide"
