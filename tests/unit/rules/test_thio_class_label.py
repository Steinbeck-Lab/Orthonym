"""Thioamides, thioureas and thiones rank with their oxygen classes, the Blue Book, "11 Amides [in
the order of the corresponding acids]";,:29585, "Ketones, pseudoketones and heterones, and their chalcogen
analogues in the order =O > =S > =Se > =Te, are senior to hydroxy compounds and their chalcogen analogues, amines, and
imines"). A name that cites one only as a 'carbamothioyl' or 'sulfanylidene' prefix on an -ol, -thiol or -nitrile parent
is no PIN (validation/spelling/checks_parent.py, principal_class_check).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_parent import principal_class_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("CNC(=S)NCCS", "2-[(methylcarbamothioyl)amino]ethane-1-thiol"),
    ("OCCN1CCNC1=S", "2-(2-sulfanylideneimidazolidin-1-yl)ethan-1-ol"),
    ("CC(=S)NCCO", "2-(ethanethioylamino)ethan-1-ol"),
    ("NC(=S)c1ccc(C#N)cc1", "4-carbamothioylbenzonitrile"),
])
def test_a_thio_class_written_as_a_prefix_is_flagged(smiles, name):
    failure = principal_class_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-41", name


@pytest.mark.parametrize("smiles,name", [
    ("CNC(=S)NCCS", "N-methyl-N'-(2-sulfanylethyl)thiourea"),
    ("OCCN1CCNC1=S", "1-(2-hydroxyethyl)imidazolidine-2-thione"),
    ("CC(=S)NCCO", "N-(2-hydroxyethyl)ethanethioamide"),
    ("OC(=O)CCNC(N)=S", "3-(carbamothioylamino)propanoic acid"),     # an acid is senior
    ("NC(=S)c1ccc(C#N)cc1", "4-cyanobenzene-1-carbothioamide"),
    ("O=C1CSC(=S)N1", "2-sulfanylidene-1,3-thiazolidin-4-one"),       # =O before =S in one class
    ("O=C1CCC(=S)CC1", "4-sulfanylidenecyclohexan-1-one"),
    ("CN(C)C(=S)SCCO", "2-hydroxyethyl dimethylcarbamodithioate"),     # an ester: not read
    ("S=C1NCCN1CCC#N", "3-(2-sulfanylideneimidazolidin-1-yl)propanenitrile"),   # a nitrile outranks the thione
])
def test_names_on_the_senior_class_pass(smiles, name):
    assert principal_class_check(Chem.MolFromSmiles(smiles), name) is None, name
