"""A ring cation nitrogen with an exocyclic anionic nitrogen (the N-imide ylides): the anion is the parent.

 "Anionic and cationic centers on different parent structures" (the Blue Book): zwitterions "may be
named by prefixing the name of the cationic center or the parts of the structure containing the cationic centers
to the name of the anionic parent structure". "CATIONIC PREFIX NAMES" (:42290), method (1) "leads to
preferred IUPAC names": "all prefix names are formed by adding the suffixes 'yl', 'ylidene', etc. to the cation
name", 'pyridin-1-ium-1-yl (preferred prefix)'. "'Ylides'" (:42499): an anionic site attached directly
to a heteroatom X+ is named "(1) as zwitterionic compounds", '2-(trimethylphosphaniumyl)propan-2-ide (PIN)'
(:42526). The anionic nitrogen is the azanide "Anionic centers on other characteristic groups",
'acetylazanide (PIN)',:41079) and the cited group goes beside the ring cation,:7272). The Blue
Book has no example of a ring N-imide, so the labels are below the PIN.

Every name is read back by a FRESH OPSIN call to the canonical SMILES of the input, charges included: the full
InChIKey does not tell the charge-separated drawings apart.
"""
import pytest
from rdkit import Chem, RDLogger

from tests.support.jars import jar_or_skip
from tests.support.rt_assert import _canon, _independent_parse, name_best_effort

RDLogger.DisableLog("rdApp.*")

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROWS = [
    # =N-: the ring cation as the '-ylidene' prefix of azanide (the CID878xxxxx series)
    ("C1=CC=C[N+]1=[N-]", "(1H-pyrrol-1-ium-1-ylidene)azanide"),
    ("CC1=CC=C(C)[N+]1=[N-]", "(2,5-dimethyl-1H-pyrrol-1-ium-1-ylidene)azanide"),
    ("C1(=CC=CC=C1)C2=CC=C(C3=CC=CC=C3)[N+]2=[N-]", "(2,5-diphenyl-1H-pyrrol-1-ium-1-ylidene)azanide"),
    ("CC(C)C1=C([N+](=[N-])C(=C1)C)C", "[2,5-dimethyl-3-(propan-2-yl)-1H-pyrrol-1-ium-1-ylidene]azanide"),
    ("CCCCC1=C([N+](=[N-])C(=C1CCCC)C2=CC(=C(C(=C2)CCC)CCC)CCC)C3=CC(=C(C(=C3)CCC)CCC)CCC",
     "[3,4-dibutyl-2,5-bis(3,4,5-tripropylphenyl)-1H-pyrrol-1-ium-1-ylidene]azanide"),
    ("C1CC[N+](C1)=[N-]", "(pyrrolidin-1-ium-1-ylidene)azanide"),
    ("C1CCC[N+](C1)=[N-]", "(piperidin-1-ium-1-ylidene)azanide"),
    # -N(-)-R: the ring cation as the '-yl' prefix, the group beside it (alphabetical order,
    ("C1=CC=[N+](C=C1)[NH-]", "(pyridin-1-ium-1-yl)azanide"),
    ("CC1=C[N+](=CC=C1)[NH-]", "(3-methylpyridin-1-ium-1-yl)azanide"),
    ("C1(=CC=CC=C1)[N-][N+]1=CC=CC=C1", "phenyl(pyridin-1-ium-1-yl)azanide"),
    ("CC(=O)[N-][N+]1=CC=CC=C1", "acetyl(pyridin-1-ium-1-yl)azanide"),
    ("CC[N-][N+]1=CC=C(C=C1)C", "ethyl(4-methylpyridin-1-ium-1-yl)azanide"),
    ("CC1=CC=C(C=C1)S(=O)(=O)[N-][N+]2=CC=CC=C2", "(4-methylbenzene-1-sulfonyl)(pyridin-1-ium-1-yl)azanide"),
    ("CC1=CC=C(C=C1)S(=O)(=O)[N-][N+]2=CC=C(S2)C3=CC=CC=C3",
     "(4-methylbenzene-1-sulfonyl)(5-phenyl-1,2-thiazol-2-ium-2-yl)azanide"),
    ("C1=CN=C[N+](=C1)[NH-]", "(pyrimidin-1-ium-1-yl)azanide"),
    # The two shapes that this builder declined before lane ION2 (a ring with an exocyclic double bond, an azole whose
    # other nitrogen carries a group). method (1), the Blue Book: "all prefix names are formed by adding
    # the suffixes 'yl', 'ylidene', etc. to the cation name", the cation name being the parent hydride and a hydron
    #,:41366), here '3H-pyrrol-1-ium' and '1-methyl-1H-imidazol-3-ium'; the source gives no ring N-imide
    # example. Independent check: the fresh OPSIN read below (2.9.0 and the dev pin both read these names) puts the
    # charges on the atoms of the input. Mutation: ``test_the_fused_cation_fallback_is_what_names_them``.
    ("CCCCC=C1C(CCCC)=C(c2ccccc2)[N+]([NH-])=C1c1ccccc1",
     "(4-butyl-3-pentylidene-2,5-diphenyl-3H-pyrrol-1-ium-1-yl)azanide"),
    ("[NH-][n+]1ccn(C)c1", "(1-methyl-1H-imidazol-3-ium-3-yl)azanide"),
]


@pytest.mark.parametrize("smiles,name", ROWS)
def test_name_is_the_azanide_with_the_ring_cation_as_prefix(smiles, name):
    jar_or_skip()
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"]) == (name, "systematic_verified"), res
    assert res["is_pin"] is False
    # an independent read: the input's own canonical SMILES, so the charges are on the input's atoms
    assert _canon(_independent_parse(name, smiles)) == _canon(smiles)


@pytest.mark.parametrize("smiles", [r[0] for r in ROWS])
def test_the_default_tier_does_not_call_it_a_pin(smiles):
    from orthonym import Orthonym
    jar_or_skip()
    res = Orthonym(style="pin").name_tiered(smiles)
    assert res["tier"] != "pin_verified" and res.get("is_pin") is not True, res


def test_the_ring_cation_prefix_is_numbered_once_with_its_ring():
    """The prefix the zwitterion builder cites: heteroatoms, the 'ium' with its free valence, then the ring's
    own prefixes in order; the double bond to the anion is the 'ylidene'."""
    from orthonym.assembly.substituent_naming import cation_to_prefix
    for smiles, prefix in [("C1=CC=[N+](C=C1)[NH-]", "pyridin-1-ium-1-yl"),
                           ("CC1=CC=C(C=C1)S(=O)(=O)[N-][N+]2=CC=C(S2)C3=CC=CC=C3", "5-phenyl-1,2-thiazol-2-ium-2-yl"),
                           ("CC1=CC=C(C)[N+]1=[N-]", "2,5-dimethyl-1H-pyrrol-1-ium-1-ylidene"),
                           ("C1CC[N+](C1)=[N-]", "pyrrolidin-1-ium-1-ylidene")]:
        mol = Chem.MolFromSmiles(smiles)
        cat = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
        an = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == -1)
        assert cation_to_prefix(mol, cat, an) == prefix


@pytest.mark.parametrize("smiles", [
    "[CH2][N-][N+]1=CC=CC=C1",                              # a radical
    "CC(=O)C1=C2C(=C([N+]2=[N-])C=C1)[N+](=O)[O-]",         # a fused ring cation and a nitro group
    "CCCCCCCCCCCCCCCCCCOC1=CC=C(C=C1)CC23CCCC4([N+]2=[N-])CC[N+](=C4)CCC3",   # a second cation
    "C1=CC=C[N+]1=[N-].[Na+]",                               # more than one fragment
])
def test_other_shapes_are_not_named_here(smiles):
    from orthonym.rules.ions import emit_ylide
    assert emit_ylide(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles", ["CCCCC=C1C(CCCC)=C(c2ccccc2)[N+]([NH-])=C1c1ccccc1", "[NH-][n+]1ccn(C)c1"])
def test_the_fused_cation_fallback_is_what_names_them(monkeypatch, smiles):
    """The mutation of the two rows above: without the name of the cation itself (``ring_cation_prefix_by_reentry``) the
    prefix of these shapes is not built and the builder names nothing, as before."""
    from orthonym.rules import ring_cation_hydro
    from orthonym.rules.ions import emit_ylide
    monkeypatch.setattr(ring_cation_hydro, "ring_cation_prefix_by_reentry", lambda *a, **k: "")
    assert emit_ylide(Chem.MolFromSmiles(smiles)) is None


def test_the_hydrazine_zwitterion_keeps_its_pin():
    """ "Amine imides" (:43023) method (1): the acyclic amine imide, based on hydrazine."""
    from orthonym import Orthonym
    jar_or_skip()
    res = Orthonym(style="pin").name_tiered("C[N-][N+](C)(C)C")
    assert (res["name"], res["tier"]) == ("1,2,2,2-tetramethylhydrazin-2-ium-1-ide", "pin_verified")


def test_a_name_that_opsin_reads_to_another_structure_is_withheld(monkeypatch):
    """The read-back is the producer's own gate: a prefix that names another group yields no name."""
    from orthonym.assembly import substituent_naming
    from orthonym.rules import charged_router
    from orthonym.rules.ions import emit_ylide
    jar_or_skip()
    mol = Chem.MolFromSmiles("CC1=CC=[N+](C=C1)[NH-]")
    cat = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
    an = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == -1)
    assert charged_router.emit_ring_cation_azanide(mol, cat, an) == "(4-methylpyridin-1-ium-1-yl)azanide"
    monkeypatch.setattr(substituent_naming, "cation_to_prefix", lambda *a, **k: "pyridin-1-ium-1-yl")
    assert charged_router.emit_ring_cation_azanide(mol, cat, an) == ""      # the methyl group is missing
    assert emit_ylide(mol) is None


def test_the_numbering_does_not_depend_on_the_drawing():
    """One species, one name: the ring read in the other direction gives the same string."""
    jar_or_skip()
    a = name_best_effort("CC(C)C1=C([N+](=[N-])C(=C1)C)C")["name"]
    b = name_best_effort("CC1=CC(C(C)C)=C(C)[N+]1=[N-]")["name"]
    assert a == b == "[2,5-dimethyl-3-(propan-2-yl)-1H-pyrrol-1-ium-1-ylidene]azanide"
