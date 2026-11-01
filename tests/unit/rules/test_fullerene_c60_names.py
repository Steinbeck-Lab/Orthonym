"""C60 fullerenes with at most one addend on a [6,6] bond, named at the best-effort tier.

Writer: ``rules.bridged_fused_pin.fullerene`` (the parent and its numberings, in ``selection.fused_split`` and
``parents.fused_parent``), with the existing hydro, indicated-hydrogen, prefix and suffix machinery of
``bridged_fused_pin`` on the numbering chooses. Certificate: ``validation.fullerene_read_back``, the engine's
own reader, because OPSIN reads no fullerene name (``carveout:fullerene``). The structures are built from the
template of ``data.fullerene_c60`` (``tests/support/fullerene_adducts.py``), so no data file is needed.

* Systematic names (the Blue Book); NUMBERING OF FULLERENES (:15336);
* Fullerenes and modified fullerenes ortho fused to organic rings or ring systems (:15464-15480):
  '3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene (PIN)', '[1,3]dioxolo[4',5':1,9](C60-Ih)[5,6]fullerene (PIN)',
  'tris(oxireno)[2',3':1,9;...](C60-Ih)[5,6]fullerene (PIN)';
* (:25834) '1-(trifluoromethyl)-1,9-dihydro(C60-Ih)[5,6]fullerene'; (:26870) '(C60-Ih)
  [5,6]fulleren-1(9H)-ol'.

No fullerene name is a PIN here (``record_non_pin_fragment``): the tier is best-effort and the label never
``pin_verified``.
"""
import random

import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.errors import OrthonymLimitError, is_failure_name
from orthonym.validation.fullerene_read_back import fullerene_read_back
from tests.support.fullerene_adducts import C60_SMILES, adduct

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

CORE = "(C60-Ih)[5,6]fullerene"


def best_effort(smiles):
    return name_compound(smiles, emit_tier="best-effort", on_failure="label")


# --- the names the book prints, and the engine's own on the same rules --------------------------------------

PINNED = [
    # (fragment on the bond 1,9 or None for the bare cage, the name)
    (None, CORE),                                                                       # (:15322)
    ("[1*]C[2*]", "3'H-cyclopropa[1,9]" + CORE),                                       # (:15464)
    ("[1*]O[2*]", "oxireno[2',3':1,9]" + CORE),                                        # the oxirene of tris(oxireno)
    ("[1*]OCO[2*]", "2'H-[1,3]dioxolo[4',5':1,9]" + CORE),                             #, with its CH2 hydrogen
    ("[1*].[2*]", "1,9-dihydro" + CORE),
    ("[1*]C(F)(F)F.[2*]", "1-(trifluoromethyl)-1,9-dihydro" + CORE),                    # (:25834)
    ("[1*]O.[2*]", "(C60-Ih)[5,6]fulleren-1(9H)-ol"),                                  # (:26870)
    ("[1*]O.[2*]O", CORE + "-1,9-diol"),
    ("[1*]C.[2*]C", "1,9-dimethyl-1,9-dihydro" + CORE),
    ("[1*]CN(C)C[2*]", "1'-methyl-1',5'-dihydro-2'H-pyrrolo[3',4':1,9]" + CORE),
    ("[1*]C(c1ccccc1)N(C)C[2*]", "1'-methyl-2'-phenyl-1',5'-dihydro-2'H-pyrrolo[3',4':1,9]" + CORE),
    ("[1*]ON=C(c1ccccc1)[2*]", "3'-phenyl[1,2]oxazolo[4',5':1,9]" + CORE),
    ("[1*]CCCC[2*]", "3',4',5',6'-tetrahydrobenzo[1,9]" + CORE),
    ("[1*]C(CCCC(=O)OC)(c1ccccc1)[2*]",
     "methyl 4-[3'-phenyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fulleren-3'-yl]butanoate"),
]


@pytest.mark.parametrize("fragment,expected", PINNED, ids=[p[1][:50] for p in PINNED])
def test_the_name_of_a_c60_adduct(fragment, expected):
    smiles = C60_SMILES if fragment is None else adduct(fragment)
    assert best_effort(smiles) == expected


def test_the_default_tier_names_it_too():
    # the default call is the cascade that ends at best-effort (``tiers.DEFAULT_TIER``)
    assert name_compound(adduct("[1*]C[2*]"), on_failure="label") == "3'H-cyclopropa[1,9]" + CORE


# --- every census class: a name, read back to the structure --------------------------------------------------

CLASSES = {
    "exocyclic 1,9-dihydro": ["[1*].[2*]", "[1*]C.[2*]", "[1*]C(F)(F)F.[2*]", "[1*]O.[2*]", "[1*]C.[2*]C",
                              "[1*]CCl.[2*]Cl", "[1*]O.[2*]O", "[1*]c1ccccc1.[2*]", "[1*]C#N.[2*]", "[1*]N.[2*]",
                              "[1*]CC(=O)O.[2*]"],
    "fused 3-ring": ["[1*]C[2*]", "[1*]C(C)(C)[2*]", "[1*]C(c1ccccc1)(C(=O)OC)[2*]",
                     "[1*]N(C)[2*]", "[1*]N(c1ccccc1)[2*]", "[1*]O[2*]", "[1*]S[2*]", "[1*][SiH2][2*]",
                     "[1*]C(C(=O)OCC)(C(=O)OCC)[2*]", "[1*]C(c1ccccc1)(CCCC(=O)O)[2*]", "[1*]C(Br)(Br)[2*]",
                     "[1*]C(c1ccc(Cl)cc1)(C#N)[2*]"],
    "fused 4-ring": ["[1*]CC[2*]", "[1*]C(C)C(C)[2*]", "[1*]C(OCC)(OCC)C(OCC)(OCC)[2*]", "[1*]CC(c1ccccc1)[2*]"],
    "fused 5-ring": ["[1*]CN(C)C[2*]", "[1*]C(c1ccccc1)N(C)C[2*]", "[1*]CNC[2*]", "[1*]ON=C(c1ccccc1)[2*]",
                     "[1*]C(c1ccccc1)C(=O)O[2*]", "[1*]C(=O)OC[2*]", "[1*]OC(C)(C)O[2*]", "[1*]N=NN(C)[2*]",
                     "[1*]CCC[2*]", "[1*]CSC[2*]", "[1*]C=CC[2*]", "[1*]N=C(C)O[2*]"],
    "fused 6-ring": ["[1*]CCCC[2*]", "[1*]CC=CC[2*]", "[1*]C(c1ccccc1)C(=O)N(C)C[2*]", "[1*]CC(c1ccccc1)N(C)C[2*]",
                     "[1*]COCC[2*]", "[1*]C=CC=C[2*]"],
}


def _cases():
    for label, fragments in CLASSES.items():
        for fragment in fragments:
            yield pytest.param(fragment, id=f"{label}: {fragment}")


@pytest.mark.parametrize("fragment", _cases())
def test_every_class_is_named_and_read_back(fragment):
    smiles = adduct(fragment)
    name = best_effort(smiles)
    assert not is_failure_name(name), name
    assert CORE.replace("fullerene", "fulleren") in name
    assert fullerene_read_back(name, smiles), name


def test_the_name_does_not_depend_on_the_atom_order_or_the_kekule_form():
    # determinism: the same structure drawn in other atom orders, and with its Kekule bonds, has one name
    smiles = adduct("[1*]C(c1ccccc1)N(C)C[2*]")
    mol = Chem.MolFromSmiles(smiles)
    names = {best_effort(smiles)}
    for seed in (3, 11):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        names.add(best_effort(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)))
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    names.add(best_effort(Chem.MolToSmiles(kek, kekuleSmiles=True)))
    assert len(names) == 1, names


def test_lowest_locants_choose_among_the_symmetries_of_the_cage():
    # (f) (the Blue Book): the group goes to the lower locant, 1 of the two atoms of the bond 1,9 (a
    # symmetry of the cage swaps them): the numbering that puts the group on any other bond is a numbering of the
    # same cage, so the bond is always 1,9 and the group always on 1
    for a, b in ((1, 9), (2, 12), (7, 8), (52, 60), (36, 37)):
        name = best_effort(adduct("[1*]C(F)(F)F.[2*]", a, b))
        assert name == "1-(trifluoromethyl)-1,9-dihydro" + CORE, (a, b, name)
    for a, b in ((1, 9), (4, 18), (21, 40)):
        assert best_effort(adduct("[1*]ON=C(c1ccccc1)[2*]", a, b)) == "3'-phenyl[1,2]oxazolo[4',5':1,9]" + CORE


def test_attached_locants_are_the_lowest_in_the_order_of_the_cage_locants():
    # (the Blue Book): "These numbers are chosen to be as low as is consistent with the numbering of
    # the compound": the heteromonocycle keeps its own numbering and the cage takes the direction that cites 4',5' (not
    # 5',4') for 1,9; an isoxazoline with the other orientation is the same compound
    one = adduct("[1*]ON=C(c1ccccc1)[2*]")
    other = adduct("[1*]C(c1ccccc1)=NO[2*]")
    assert best_effort(one) == best_effort(other)


# --- the tiers ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tier", ["pin", "valid", "complete"])
def test_a_narrower_tier_declines_the_cage_at_once(tier):
    smiles = adduct("[1*]C[2*]")
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, emit_tier=tier, raise_on_limit=True)
    assert err.value.code == "FULLERENE"
    assert is_failure_name(name_compound(smiles, emit_tier=tier, on_failure="label"))


def test_the_label_is_best_effort_and_never_a_pin():
    from orthonym.cli import _emit_tier_flags
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    for fragment in ("[1*]C[2*]", "[1*]C(F)(F)F.[2*]", "[1*]O.[2*]", "[1*]CN(C)C[2*]"):
        row = namer.name_tiered(adduct(fragment))
        assert row["tier"] == "best_effort" and row["is_pin"] is False, row
        assert row["gate_outcome"] == "carveout:fullerene", row
    row = namer.name_tiered(C60_SMILES)
    assert row["name"] == CORE and row["tier"] == "best_effort" and row["is_pin"] is False


def test_the_default_tier_never_ships_a_fullerene_name_as_a_pin():
    from orthonym.cli import _emit_tier_flags
    namer = Orthonym(style="pin", **_emit_tier_flags("pin"))
    with pytest.raises(OrthonymLimitError):
        namer.name(C60_SMILES, raise_on_limit=True)


# --- outside the scope: the fast decline stays -------------------------------------------------------------

OUT_OF_SCOPE = {
    "adduct on a [5,6] bond": lambda: adduct("[1*]C[2*]", 1, 2),
    "ring of eight atoms": lambda: adduct("[1*]CCCCCC[2*]"),
    "fused ring that is bicyclic": lambda: adduct("[1*]c1ccccc1C[2*]"),
    "stereo layer": lambda: adduct("[1*][C@H](C)N(C)C[2*]"),
    "charged addend": lambda: adduct("[1*]C[N+](C)(C)C[2*]"),
    "second fragment": lambda: adduct("[1*]C[2*]") + ".O",
}


@pytest.mark.parametrize("label", sorted(OUT_OF_SCOPE))
def test_out_of_scope_rows_still_decline_fast(label):
    import time
    smiles = OUT_OF_SCOPE[label]()
    start = time.time()
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, emit_tier="best-effort", raise_on_limit=True)
    assert err.value.code == "FULLERENE"
    assert time.time() - start < 8.0


def test_two_addends_on_the_cage_decline():
    # a cyclopropane on 1,9 and another on 2,12: the ring system holds two attached rings, so it is not read
    from orthonym.data.fullerene_c60 import C60_MAPPED_SMILES
    from orthonym.perception.fullerene import c60_adduct
    both = Chem.RWMol(Chem.MolFromSmiles(C60_MAPPED_SMILES, sanitize=False))
    where = {a.GetAtomMapNum(): a.GetIdx() for a in both.GetAtoms()}
    for a, b in ((1, 9), (2, 12)):
        both.GetBondBetweenAtoms(where[a], where[b]).SetBondType(Chem.BondType.SINGLE)
        carbon = both.AddAtom(Chem.Atom(6))
        both.AddBond(carbon, where[a], Chem.BondType.SINGLE)
        both.AddBond(carbon, where[b], Chem.BondType.SINGLE)
    for atom in both.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetNoImplicit(False)
    mol = both.GetMol()
    Chem.SanitizeMol(mol)
    assert c60_adduct(mol) is None
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(Chem.MolToSmiles(mol), emit_tier="best-effort", raise_on_limit=True)
    assert err.value.code == "FULLERENE"


def test_other_cages_still_decline_at_best_effort():
    c70 = ("c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c%12c%13c%14c%15c%16c%17c%14c%14c%18c%13c%11c1c1c5c%10"
           "c5c(c%14c%10c%17c%11c%13c%16c%14c%16c%15c%12c%12c%16c(c2c7c%126)c2c8c3c(c%13c%142)c2c9c4c5c%10c%112)c%181")
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(c70, emit_tier="best-effort", raise_on_limit=True)
    assert err.value.code == "FULLERENE"


# --- perception and the natural-product detector --------------------------------------------------------------

def test_perception_reads_the_scope():
    from orthonym.perception.fullerene import c60_adduct
    bare = c60_adduct(Chem.MolFromSmiles(C60_SMILES))
    assert bare is not None and bare.sp3 == () and bare.ring == () and len(bare.locants) == 120
    exo = c60_adduct(Chem.MolFromSmiles(adduct("[1*]O.[2*]")))
    assert exo is not None and len(exo.sp3) == 2 and exo.ring == () and len(exo.locants) == 4
    assert all({m[a] for a in exo.sp3} == {1, 9} for m in exo.locants)
    fused = c60_adduct(Chem.MolFromSmiles(adduct("[1*]CN(C)C[2*]")))
    assert fused is not None and len(fused.ring) == 3 and len(fused.locants) == 4


def test_a_cage_is_no_natural_product_scaffold():
    # matching the scaffold table against a cage cost 3.4 s; a molecule with a cage is named by the fullerene rules
    import time

    from orthonym.perception.natural_products import _detect_natural_product_impl
    mol = Chem.MolFromSmiles(adduct("[1*]CN(C)C[2*]"))
    start = time.time()
    assert _detect_natural_product_impl(mol) is None
    assert time.time() - start < 1.0


def test_the_porphyrin_core_is_not_searched_inside_a_cage():
    # the 24-atom core takes seconds to match against 31 rings; a molecule with a cage is matched without it, and the
    # matches of any other molecule are the same as before
    import time

    from orthonym.rules.porphyrin import _core_matches
    tpp = Chem.MolFromSmiles("c1ccc(cc1)-c1c2ccc(n2)c(-c2ccccc2)c2ccc([nH]2)c(-c2ccccc2)c2ccc(n2)c(-c2ccccc2)c2ccc1[nH]2")
    assert len(_core_matches(tpp)) == 8
    cage = Chem.MolFromSmiles(adduct("[1*]C(C(N)N)C(C(N)N)[2*]"))     # four nitrogen atoms, no core
    start = time.time()
    assert _core_matches(cage) == []
    assert time.time() - start < 1.0


def test_a_stereo_row_is_not_named_flat_and_offered():
    # the offer that names the stereo-free structure and composes the stereo back cannot read a fullerene name back:
    # a stereo row declines at once instead of after a second naming
    import time
    smiles = adduct("[1*][C@H](CC)N(C)C[2*]")
    start = time.time()
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, emit_tier="best-effort", raise_on_limit=True)
    assert err.value.code == "FULLERENE" and time.time() - start < 5.0
    assert is_failure_name(name_compound(smiles, emit_tier="best-effort", on_failure="label"))


def test_the_gate_records_the_non_pin_part_where_the_name_ships():
    from orthonym.data.fullerene_c60 import C60_CORE_TOKEN
    from orthonym.metrics import provenance as pv
    from orthonym.metrics.provenance import best_effort_request_ctx
    from orthonym.validation.fullerene_read_back import fullerene_gate_ships
    smiles = adduct("[1*]C[2*]")
    name = "3'H-cyclopropa[1,9]" + CORE
    token = best_effort_request_ctx.set(True)
    try:
        assert fullerene_gate_ships(name, smiles)
        assert C60_CORE_TOKEN in pv.get_provenance()["non_pin_fragments"]
        assert not fullerene_gate_ships(name, adduct("[1*]CC[2*]"))
    finally:
        best_effort_request_ctx.reset(token)
    assert not fullerene_gate_ships(name, smiles)                    # outside a best-effort request nothing ships


def test_a_name_built_by_the_decomposition_route_is_labelled_like_the_others():
    # the decomposition route claims the OPSIN reading of the parts it names; no OPSIN reads the whole fullerene name,
    # so the row is the carve-out's, best-effort and unverified by OPSIN, never 'systematic_verified'
    from orthonym.cli import _emit_tier_flags
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    row = namer.name_tiered(adduct("[1*]C(C(=O)OC)N(C(=O)CCCCCCCCC(=O)N1CCOCCOCCOCCOCCOCC1)C(C(=O)OC)[2*]"))
    assert row["name"] and "fulleren" in row["name"], row
    assert row["tier"] == "best_effort" and row["is_pin"] is False and row["opsin"] == "unverified", row
    assert row["gate_outcome"] == "carveout:fullerene"
