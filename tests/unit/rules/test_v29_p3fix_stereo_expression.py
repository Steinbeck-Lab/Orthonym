"""v29 P3-FIX Item 1 — the widened organyl prefix must EXPRESS its stereo or REFUSE.

The Phase 3 organyl migration widened ``rules.substituent_purity.organyl_prefix_name``
from a hydrocarbon-only walker to the shared 5-tier chokepoint.  The retired walker
refused every unsaturated fragment; the chokepoint names them -- but
``_add_substituent_stereo`` reads only **atom** ``_CIPCode``, never **bond**
``_CIPCode`` (its own sibling says so verbatim at
``substituent_naming.py:150-152``), so an E/Z double bond inside the fragment was
silently dropped.  Both geometric isomers of ``C/C=C/[AsH2]`` therefore emitted the
single name ``(prop-1-en-1-yl)arsane`` -- two distinct compounds, one name, and
neither round-trips.

This module pins BOTH halves of the repair:

**(a) express what is derivable.**  ``_name_unsaturated_chain`` already derives the
substituent's own numbering (free valence lowest, P-31.1.3.4), so it -- and only it
-- can locate the descriptor.  The located form is the verbatim Blue Book PIN:

* ``## **P-91.3** NAMING OF STEREOISOMERS`` (``BlueBookV2.md:44686``):
  ``(5Z)-4-[(1E)-prop-1-en-1-yl]hepta-1,5-diene (PIN)`` -- the token
  ``(1E)-prop-1-en-1-yl`` verbatim.
* ``### Example 2`` under ``## **P-91.3** NAMING OF STEREOISOMERS``
  (``BlueBookV2.md:45437``):
  ``(2Z,5Z,7S,11Z)-5-[(2E)-but-2-en-1-yl]-9-[(2Z)-but-2-en-1-yl]trideca-2,5,8,11-tetraen-7-ol (PIN)``
  -- the token ``(2E)-but-2-en-1-yl`` verbatim.
* ``## **P-46.3** PRINCIPAL SUBSTITUENT CHAINS IN COMPOUNDS WITH STEREOGENIC CENTERS``
  (``BlueBookV2.md:23014``): ``[(2Z,4R,5E)-4-methylhepta-2,5-dien-4-yl]siline (PIN)``
  -- a MIXED R/S + E/Z located block on an alkenyl organyl prefix, cited on a
  mononuclear Group-14 parent hydride: exactly the class this phase widened.  It
  also fixes the citation ORDER question: descriptors run in LOCANT order (2,4,5),
  not R/S-before-E/Z.
* ``### P-91.2.1.2.1 Stereodescriptors used in substitutive nomenclature``
  (``BlueBookV2.md:44624``): "*In preferred IUPAC names, stereodescriptors,
  preceded by a locant, **must be** cited to specify each stereogenic unit*" --
  so the locant is REQUIRED, not optional.
* ``## **P-92.1.1** Stereogenic units`` (``BlueBookV2.md:44718``): "*In preferred
  IUPAC names, all stereogenic units must be specified, unless an omission is
  allowed according to P-91.2.2.*"  ``### **P-91.2.2** Omission of
  stereodescriptors`` (``BlueBookV2.md:44635``) licenses omission ONLY for 3-7
  membered unsaturated alicyclics, the 8-ring carve-out, von Baeyer, spiro, fused,
  cyclophane and ring assemblies -- an acyclic side chain is in none of them.

**(b) prove the rest, and fail closed.**  Everything the emitters cannot locate
(multi-centre acyclic R/S, whose locants the D-09 rule forbids fabricating) must
make ``organyl_prefix_name`` return None rather than ship a stereo-stripped name.
The obligation is a COUNT identity, not a pattern: the name must express exactly as
many descriptor tokens as the fragment defines stereo elements.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.rules.substituent_purity import organyl_prefix_name


def _organyl(smiles, hub_symbol='As'):
    """``organyl_prefix_name`` for the LARGEST organyl on the single hub atom."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    assign_stereochemistry(mol)
    hubs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == hub_symbol]
    assert len(hubs) == 1, smiles
    hub = hubs[0]
    best = None
    for nbr in mol.GetAtomWithIdx(hub).GetNeighbors():
        if nbr.GetSymbol() != 'C':
            continue
        from orthonym.rules.substituent_purity import _fragment_atoms
        frag = _fragment_atoms(mol, nbr.GetIdx(), hub)
        if frag and (best is None or len(frag) > best[1]):
            best = (nbr.GetIdx(), len(frag))
    assert best is not None, smiles
    return organyl_prefix_name(mol, best[0], hub)


@pytest.fixture
def ungated_namer(monkeypatch):
    """A namer with the SELF-01 OPSIN validity gate explicitly DISABLED, so the
    producer's own output is asserted with nothing downstream able to rescue it."""
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)
    return Orthonym()


# --------------------------------------------------------------------------
# (a) E/Z is EXPRESSED, in the BB-verbatim located form
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # BB 44686 verbatim token, both geometries.
    ("C/C=C/[AsH2]",  "(1E)-prop-1-en-1-yl"),
    ("C/C=C\\[AsH2]", "(1Z)-prop-1-en-1-yl"),
    # BB 45437 verbatim token, both geometries.
    ("C/C=C/C[AsH2]",  "(2E)-but-2-en-1-yl"),
    ("C/C=C\\C[AsH2]", "(2Z)-but-2-en-1-yl"),
    # Internal attachment: the free valence still takes the lowest locant
    # (P-31.1.3.4), so the ene locant is 3 and the descriptor rides on it.
    ("CC(/C=C/C)[AsH2]",  "(3E)-pent-3-en-2-yl"),
    ("CC(/C=C\\C)[AsH2]", "(3Z)-pent-3-en-2-yl"),
    # Two stereogenic double bonds -> one comma-joined block in locant order.
    ("C/C=C/C=C/[AsH2]",  "(1E,3E)-penta-1,3-dien-1-yl"),
    ("C/C=C\\C=C/[AsH2]", "(1Z,3Z)-penta-1,3-dien-1-yl"),
    ("C/C=C/C=C\\[AsH2]", "(1Z,3E)-penta-1,3-dien-1-yl"),
])
def test_ez_double_bond_is_expressed_with_its_locant(smiles, expected):
    assert _organyl(smiles) == expected


def test_the_two_geometric_isomers_never_share_a_name():
    """The defect in one assertion: E and Z are different compounds."""
    e = _organyl("C/C=C/[AsH2]")
    z = _organyl("C/C=C\\[AsH2]")
    assert e is not None and z is not None
    assert e != z


@pytest.mark.parametrize("smiles,expected", [
    # BB 23014's shape: R/S and E/Z in ONE block, cited in LOCANT order.
    ("C/C=C/[C@@H](C)[AsH2]", "(2R,3E)-pent-3-en-2-yl"),
    ("C/C=C/[C@H](C)[AsH2]",  "(2S,3E)-pent-3-en-2-yl"),
])
def test_mixed_rs_and_ez_share_one_locant_ordered_block(smiles, expected):
    assert _organyl(smiles) == expected


# --------------------------------------------------------------------------
# (b) what cannot be located FAILS CLOSED
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    # Two stereocentres in one acyclic fragment: `_add_substituent_stereo`'s
    # multi-centre branch returns the name UNCHANGED (D-09 forbids fabricating
    # the locants), so both diastereomers used to emit `3-methylpentan-2-yl`.
    "CC[C@@H](C)[C@@H](C)[As](C)C",
    "CC[C@@H](C)[C@H](C)[As](C)C",
])
def test_unexpressible_multicentre_stereo_fails_closed(smiles):
    assert _organyl(smiles) is None


def test_the_multicentre_diastereomers_no_longer_collide():
    """Before the fix both returned the identical stereo-free name."""
    a = _organyl("CC[C@@H](C)[C@@H](C)[As](C)C")
    b = _organyl("CC[C@@H](C)[C@H](C)[As](C)C")
    assert not (a is not None and a == b)


# --------------------------------------------------------------------------
# (c) no collateral damage: already-correct classes are byte-identical
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # Achiral unsaturation: no stereo element, so no descriptor may appear.
    ("C=C[AsH2]",        "ethenyl"),
    ("C#C[AsH2]",        "ethynyl"),
    ("C=CC[AsH2]",       "prop-2-en-1-yl"),
    # Undefined geometry (no /\ in the SMILES) must stay descriptor-free: a PIN
    # may not invent a configuration the structure does not carry.
    ("CC=C[AsH2]",       "prop-1-en-1-yl"),
    # Saturated R/S at the attachment: the pre-existing located form, unchanged.
    ("CC[C@@H](C)[As](C)C", "(2R)-butan-2-yl"),
    ("CC[C@H](C)[As](C)C",  "(2S)-butan-2-yl"),
    # Ring stereo was ALREADY expressed by the ring path -- must stay so.
    ("C[C@H]1CC[C@@H](C)CC1[AsH2]", "(2S,5R)-2,5-dimethylcyclohexyl"),
    ("C[C@H]1CC[C@H](C)CC1[AsH2]",  "(2S,5S)-2,5-dimethylcyclohexyl"),
    # Plain hydrocarbons: the migration's own class, no stereo anywhere.
    ("CCC[AsH2]",        "propyl"),
    ("c1ccccc1[AsH2]",   "phenyl"),
    ("C1CCCCC1[AsH2]",   "cyclohexyl"),
])
def test_nonstereo_and_already_expressed_classes_are_unchanged(smiles, expected):
    assert _organyl(smiles) == expected


# --------------------------------------------------------------------------
# (d) end to end, with the producer's answer reaching the shipped name
# --------------------------------------------------------------------------

@pytest.mark.parametrize("e_smiles,z_smiles", [
    ("C/C=C/[Si](Cl)(Cl)Cl", "C/C=C\\[Si](Cl)(Cl)Cl"),
    ("C/C=C/NN",             "C/C=C\\NN"),
    ("C/C=C/C[AsH2]",        "C/C=C\\C[AsH2]"),
])
def test_end_to_end_geometric_isomers_get_different_names(
        ungated_namer, e_smiles, z_smiles):
    e = ungated_namer.name(Chem.CanonSmiles(e_smiles))
    z = ungated_namer.name(Chem.CanonSmiles(z_smiles))
    assert e != z, (e, z)


# --------------------------------------------------------------------------
# (e) the BRANCHED alkenyl producer locates its descriptor from the same chain
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # A branched alkenyl: `_name_unsaturated_chain` declines (its `c_nbrs > 2`
    # guard), so `_name_branched_alkenyl_substituent` owns these. Before the fix
    # they were the 86% majority of the fragments the stereo obligation refused.
    ("CC(C)/C=C(C)/[AsH2]",  "(2E)-4-methylpent-2-en-2-yl"),
    ("CC(C)/C=C(C)\\[AsH2]", "(2Z)-4-methylpent-2-en-2-yl"),
])
def test_branched_alkenyl_expresses_its_geometry(smiles, expected):
    assert _organyl(smiles) == expected


def test_branched_alkenyl_geometric_isomers_never_share_a_name():
    assert _organyl("CC(C)/C=C(C)/[AsH2]") != _organyl("CC(C)/C=C(C)\\[AsH2]")


# --------------------------------------------------------------------------
# (f) the CONSTITUTION guard the stereo obligation surfaced
# --------------------------------------------------------------------------
# `_name_saturated_substituted_chain` is named, and documented, for a SATURATED
# halogenated chain, but nothing enforced it: an unsaturated fragment was named on
# the alkane stem and the double bond DISAPPEARED. That is a different compound,
# not a stereo-underspecified one. Pinned here because the stereo-count obligation
# only catches it while the lost bond happens to carry DEFINED geometry -- these
# rows must keep holding when it does not.

@pytest.mark.parametrize("smiles,start,excl,forbidden", [
    # -CH2-CH=CH-Cl is prop-2-en-1-yl, NOT propyl.
    ("N[C@@H](C/C=C\\Cl)C(=O)O", 2, 1, "3-chloropropyl"),
    # -CH=CH-Cl is ethenyl, NOT ethyl.
    ("N[C@@H](C/C=C\\Cl)C(=O)O", 3, 2, "2-chloroethyl"),
    # ...and with the geometry UNDEFINED, where no stereo check can help.
    ("NC(CC=CCl)C(=O)O", 2, 1, "3-chloropropyl"),
    ("NC(CC=CCl)C(=O)O", 3, 2, "2-chloroethyl"),
])
def test_halogenated_chain_namer_never_drops_a_double_bond(
        smiles, start, excl, forbidden):
    from orthonym.assembly.substituent_naming import (
        _name_saturated_substituted_chain)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assign_stereochemistry(mol)
    from orthonym.rules.substituent_purity import _fragment_atoms
    frag = _fragment_atoms(mol, start, excl)
    assert frag is not None
    # the saturated namer must DECLINE an unsaturated fragment outright
    assert _name_saturated_substituted_chain(mol, frag, start, set()) is None
    # and whatever the cascade finally says must not be the alkane spelling
    from orthonym.assembly.substituent_enumerator import name_substituent
    got = name_substituent(mol, frag, start, allow_mancude=False)
    assert got != forbidden, got


@pytest.mark.parametrize("smiles,start,excl,expected", [
    # The saturated class it DOES own stays byte-identical.
    ("NC(CCCCl)C(=O)O", 2, 1, "3-chloropropyl"),
    ("NC(CCCl)C(=O)O",  2, 1, "2-chloroethyl"),
])
def test_saturated_halogenated_chain_is_unchanged(smiles, start, excl, expected):
    from orthonym.assembly.substituent_naming import (
        _name_saturated_substituted_chain)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assign_stereochemistry(mol)
    from orthonym.rules.substituent_purity import _fragment_atoms
    frag = _fragment_atoms(mol, start, excl)
    assert _name_saturated_substituted_chain(mol, frag, start, set()) == expected


def test_removing_the_saturation_guard_really_does_delete_the_double_bond():
    """MUTATION PROOF of the guard above — the reproduction nobody had done.

    The two tests above assert only that the guarded function DECLINES.  That is
    not evidence the guard is load-bearing: a function that declined for some
    other reason would pass them identically.  This test removes the saturation
    decline from the live source and shows the atom/bond loss appear.

    Measured here:

        fragment          guarded (HEAD)   guard removed
        -CH2-CH=CH-Cl     None             '3-chloropropyl'
        -CH=CH-Cl         None             '2-chloroethyl'
        -CH2CH2CH2-Cl     '3-chloropropyl' '3-chloropropyl'   <- CONTROL

    The control is what makes it a collision rather than merely a wrong name:
    `3-chloropropyl` is the CORRECT name of the saturated chain, so without the
    guard two different constitutions received one name.

    The mutation target is asserted present before anything is measured, so a
    drifted source cannot make this test pass by mutating nothing.
    """
    import inspect

    from orthonym.assembly import substituent_naming as sn

    src = inspect.getsource(sn._name_saturated_substituted_chain)
    decline = (
        "    # SATURATED is in this function's name and contract: decline otherwise.\n"
        "    for _b in mol.GetBonds():\n"
        "        if (_b.GetBeginAtomIdx() in sub_set and _b.GetEndAtomIdx() in sub_set\n"
        "                and _b.GetBondTypeAsDouble() != 1.0):\n"
        "            return None\n"
    )
    assert decline in src, (
        "the saturation decline is not where this test expects it; re-derive the "
        "mutation target rather than deleting this test"
    )
    namespace = dict(sn.__dict__)
    exec(compile(src.replace(decline, "").replace(
        "def _name_saturated_substituted_chain", "def _unguarded", 1),
        "<mutant>", "exec"), namespace)
    unguarded = namespace["_unguarded"]

    from orthonym.rules.substituent_purity import _fragment_atoms

    def frag_of(smiles, start, excl):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assign_stereochemistry(mol)
        atoms = _fragment_atoms(mol, start, excl)
        assert atoms is not None
        return mol, atoms

    # unsaturated: the guard is the only thing standing between us and the loss
    for smiles, start, excl, lost_name in (
        ("NC(CC=CCl)C(=O)O", 2, 1, "3-chloropropyl"),
        ("NC(CC=CCl)C(=O)O", 3, 2, "2-chloroethyl"),
    ):
        mol, atoms = frag_of(smiles, start, excl)
        assert sn._name_saturated_substituted_chain(mol, atoms, start, set()) is None
        assert unguarded(mol, atoms, start, set()) == lost_name, (
            "removing the guard must reproduce the alkane spelling; if it does "
            "not, the guard is not what protects this class"
        )

    # CONTROL: the saturated class is byte-identical with and without the guard,
    # so the guard is not merely refusing everything.
    mol, atoms = frag_of("NC(CCCCl)C(=O)O", 2, 1)
    assert sn._name_saturated_substituted_chain(mol, atoms, 2, set()) == \
        "3-chloropropyl"
    assert unguarded(mol, atoms, 2, set()) == "3-chloropropyl"


# --------------------------------------------------------------------------
# (g) the counting contract itself — three properties no naming witness reaches
# --------------------------------------------------------------------------

def test_stereo_outside_the_fragment_is_not_the_prefix_obligation():
    """A prefix answers for ITS OWN fragment only.

    `C/C=C/[As](C)C` carries one E/Z bond, but it is in the propenyl arm. If the
    counter forgot to check bond MEMBERSHIP, that bond would be charged to the two
    methyls as well, whose names can never express it, and `methyl` would refuse.
    """
    from orthonym.rules.stereochemistry import count_defined_stereo_in_fragment
    from orthonym.rules.substituent_purity import _fragment_atoms
    mol = Chem.MolFromSmiles("C/C=C/[As](C)C")
    assign_stereochemistry(mol)
    hub = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'As'][0]
    seen = {}
    for nbr in mol.GetAtomWithIdx(hub).GetNeighbors():
        frag = _fragment_atoms(mol, nbr.GetIdx(), hub)
        seen[nbr.GetIdx()] = (count_defined_stereo_in_fragment(mol, frag),
                              organyl_prefix_name(mol, nbr.GetIdx(), hub))
    assert sorted(v[0] for v in seen.values()) == [0, 0, 1], seen   # NOT [1, 1, 1]
    assert sorted(v[1] for v in seen.values()) == [
        '(1E)-prop-1-en-1-yl', 'methyl', 'methyl'], seen


def test_p91_2_2_small_ring_double_bond_is_not_a_demanded_descriptor():
    """A <8-membered ring double bond is ring-strain-fixed, so P-91.2.2 recommends
    OMITTING its descriptor — it must not be counted as an obligation the prefix
    failed to meet.

    `### **P-91.2.2** Omission of stereodescriptors` (`BlueBookV2.md:44635`):
    "*The omission of stereodescriptors specifying double bonds is recommended in
    the case of three- through seven-membered unsaturated alicyclic compounds where
    any double bond has a fixed configuration*".

    RDKit does not currently put a `_CIPCode` on such a bond, so no SMILES reaches
    this branch; the label is therefore set explicitly. That is the point — the
    carve-out must keep holding if RDKit's labelling ever widens.
    """
    from orthonym.rules.stereochemistry import count_defined_stereo_in_fragment
    from orthonym.rules.substituent_purity import _fragment_atoms
    mol = Chem.MolFromSmiles("C1=CCCCC1[AsH2]")          # cyclohex-2-en-1-yl
    assign_stereochemistry(mol)
    hub = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'As'][0]
    nbr = [n.GetIdx() for n in mol.GetAtomWithIdx(hub).GetNeighbors()
           if n.GetSymbol() == 'C'][0]
    frag = _fragment_atoms(mol, nbr, hub)
    ring_dbl = [b for b in mol.GetBonds()
                if b.GetBondTypeAsDouble() == 2.0 and b.IsInRing()]
    assert len(ring_dbl) == 1
    ring_dbl[0].SetProp('_CIPCode', 'Z')
    assert count_defined_stereo_in_fragment(mol, frag) == 0
    assert organyl_prefix_name(mol, nbr, hub) == 'cyclohex-2-en-1-yl'


def test_over_expressed_stereo_also_fails_closed(monkeypatch):
    """The obligation is an IDENTITY, not a lower bound.

    A name citing MORE stereo than the structure defines is an attribution error —
    it specifies a configuration at a position the input leaves open — so it must
    refuse just as under-expression does. No producer emits such a name today, so
    the chokepoint is stubbed to return one.
    """
    import orthonym.assembly.substituent_enumerator as se
    mol = Chem.MolFromSmiles("CC[C@@H](C)[As](C)C")       # exactly ONE R/S centre
    assign_stereochemistry(mol)
    hub = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'As'][0]
    start = [n.GetIdx() for n in mol.GetAtomWithIdx(hub).GetNeighbors()
             if n.GetSymbol() == 'C' and n.GetDegree() > 1][0]
    assert organyl_prefix_name(mol, start, hub) == '(2R)-butan-2-yl'

    monkeypatch.setattr(se, 'name_substituent',
                        lambda *a, **k: '(2R,3S)-butan-2-yl')
    assert organyl_prefix_name(mol, start, hub) is None
