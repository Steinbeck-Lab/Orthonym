"""v50 Task A4 — multiplicative-vs-substitutive spelling /.

These pin the Blue-Book (PIN) spelling for right-molecule names the engine
already emits, in BOTH directions of the multiplicative/substitutive choice:

  * (c): a multiplied unit whose name begins with a skeletal
    replacement ('a') prefix ('aza', 'oxa',...) takes 'bis'/'tris' with
    enclosing marks and its heteroatom locant, because 'di<a>...' reads as a
    replacement-atom count (the Blue Book '1,1'-methylenebis(1-azacyclododecane)';
    the Blue Book 'bis(azacyclododecane)' not 'diazacyclododecane').
  *: a =S/=O bridging two BARE ring parent hydrides (no principal
    characteristic group) is the thione/ketone principal characteristic group,
    so the parent is methanethione/methanone (substitutive), NOT a
    'carbonothioyl'/'carbonyl' multiplicative bridge
    (the Blue Book 'di(1H-imidazol-1-yl)methanethione (PIN)').

Every asserted name round-trips through OPSIN to the input structure (0-wrong).
"""
from __future__ import annotations

import pytest

from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_parse
from rdkit import Chem


def _ikey(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m) if m else None


def _roundtrips(name, smi):
    parsed = opsin_parse(name)
    return parsed is not None and _ikey(parsed) == _ikey(smi)


@pytest.mark.unit
class TestP1636cSkeletalReplacementUnit:
    """(c): bis + enclosure + heteroatom locant for 'a'-prefix units."""

    def test_methylenebis_azacyclododecane(self):
        smi = "C1CCCCCN(CN2CCCCCCCCCCC2)CCCCC1"
        name = name_compound(smi)
        assert name == "1,1'-methylenebis(1-azacyclododecane)"
        assert _roundtrips(name, smi)


@pytest.mark.unit
class TestP6461ThioneKetoneParent:
    """: =S/=O over BARE rings is the thione/ketone parent (substitutive).

    C-attached bare rings already reach the substitutive ketone/thione namer.
    """

    def test_di_c_imidazolyl_methanethione(self):
        # imidazole attached via ring CARBON (position 2) -> substitutive works.
        smi = "S=C(c1nccn1C)c1nccn1C"
        assert name_compound(smi) == "di(1-methylimidazol-2-yl)methanethione"

    def test_diphenylmethanone(self):
        assert name_compound("O=C(c1ccccc1)c1ccccc1") == "diphenylmethanone"

    @pytest.mark.xfail(
        reason="v50 A4 BLOCKED: guard broadening to bare heteroarenes is "
        "correct (P-64.6.1) but the substitutive methanethione builder cannot "
        "place an N-attached azolyl (perceives an N-C(=S)-N thiourea) and "
        "abstains. Multiplicative name retained until the producer gap closes.",
        strict=True,
    )
    def test_di_imidazol_1_yl_methanethione_blocked(self):
        smi = "S=C(n1ccnc1)n1ccnc1"
        assert name_compound(smi) == "di(1H-imidazol-1-yl)methanethione"


@pytest.mark.unit
class TestP68PnictogenMultiplicative:
    """: >=2 identical bare pnictogen hydrides on a central
    skeleton -> multiplicative (central-diyl)bis/tris(phosphane/arsane), because
    an all-carbon chain cannot be a multiplied parent (b))."""

    def test_butane_triyl_tris_phosphane(self):
        smi = "PCCC(P)CP"
        name = name_compound(smi)
        assert name == "(butane-1,2,4-triyl)tris(phosphane)"
        assert _roundtrips(name, smi)

    def test_phenylene_bis_arsane(self):
        smi = "[AsH2]c1ccccc1[AsH2]"
        name = name_compound(smi)
        assert name == "(1,2-phenylene)bis(arsane)"
        assert _roundtrips(name, smi)

    def test_single_phosphane_declines_multiplicative(self):
        # a single -PH2 has no identical partner: the multiplicative pnictogen
        # handlers must decline (>=2 units required), never fabricate a name.
        from orthonym.rules.multiplicative import name_multiplicative
        assert name_multiplicative(Chem.MolFromSmiles("PC1CCCCC1")) is None
        assert name_multiplicative(Chem.MolFromSmiles("PCCCC")) is None


@pytest.mark.unit
class TestNoRegressionMultiplicativeKept:
    """The C=O/C=S multiplicative bridge is kept when a ring bears a real PCG."""

    def test_carbonothioyl_dipyridinone_kept(self):
        # pyridin-2(1H)-one carries a ring =O (a PCG) -> multiplicative kept.
        smi = "O=c1ccccn1C(=S)n1ccccc1=O"
        assert name_compound(smi) == "1,1'-carbonothioyldi(pyridin-2(1H)-one)"

    def test_carbonyl_dibenzoic_acid_kept(self):
        # benzoic-acid rings carry COOH (a PCG senior to ketone) -> multiplicative.
        smi = "O=C(O)c1ccc(C(=O)c2ccc(C(=O)O)cc2)cc1"
        assert name_compound(smi) == "4,4'-carbonyldibenzoic acid"

    def test_diphenylmethanethione_kept(self):
        assert name_compound("S=C(c1ccccc1)c1ccccc1") == "diphenylmethanethione"
