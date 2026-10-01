"""Hydrogenation state of a bridged fused parent: (:14245),
(:14583), (:14642), (:16880), (:17026)."""
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import hydro, numbering, prefixes
from orthonym.rules.bridged_fused_pin.selection import best_splits, ring_system


def _state(smiles):
    mol = Chem.MolFromSmiles(smiles)
    splits = best_splits(mol, ring_system(mol))
    split = max(splits, key=lambda s: s.n_double)
    names = [prefixes.bridge_prefix(mol, split, i).name for i in range(len(split.bridges))]
    nums = numbering.numberings(mol, split, names)
    state = hydro.hydro_state(mol, split)

    def key(n):   # (a), (b), then (b) and (e) as build orders them
        ih, hyd = hydro.choose_indicated_hydrogen(state, n.atom_to_locant)
        return (n.attachment_set, n.citation_order,
                numbering.locant_tuple(n.atom_to_locant[a] for a in ih),
                numbering.locant_tuple(n.atom_to_locant[a] for a in hyd))
    return mol, split, state, min(nums, key=key).atom_to_locant


def test_bridge_on_a_fusion_bond_leaves_4a_and_8a_without_a_double_bond():
    # 4a,8a-ethanonaphthalene (:14249): 4a and 8a have four ring bonds
    _, _, state, a2l = _state("C1=CC23C=CC=CC2(C=C1)CC3")
    assert sorted(a2l[a] for a in state.eligible) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert (state.mancude_double, state.n_double, state.ih_sets) == (4, 4, (frozenset(),))


def test_indicated_hydrogen_takes_the_lowest_locant():
    # 2,4a-methano: nine atoms can carry a double bond -> one indicated hydrogen;
    # it may sit at 2 or 5, gives it the lower locant (2H), hydro takes 1,5
    _, _, state, a2l = _state("CC1(C)C2=CC=CC(C)(C)C23C=CC1C3")
    assert sorted(sorted(a2l[a] for a in s) for s in state.ih_sets) == [[2], [5]]
    ih, hyd = hydro.choose_indicated_hydrogen(state, a2l)
    assert hydro.ih_text(ih, a2l) == "2H"
    assert hydro.hydro_text(state, hyd, a2l) == "1,5-dihydro"


def test_total_hydrogenation_omits_the_locants():
    _, _, state, a2l = _state("C1CC2CC1C1CCCCC12")
    ih, hyd = hydro.choose_indicated_hydrogen(state, a2l)
    assert hydro.hydro_text(state, hyd, a2l) == "decahydro"
    _, _, state, a2l = _state("C1CC23CCCCC2(CC1)CC3")
    ih, hyd = hydro.choose_indicated_hydrogen(state, a2l)
    assert hydro.hydro_text(state, hyd, a2l) == "octahydro"


def test_mancude_bridged_parent_has_no_hydro_prefix():
    _, _, state, a2l = _state("C1=C2CC(=C1)c1ccccc12")
    ih, hyd = hydro.choose_indicated_hydrogen(state, a2l)
    assert (hydro.hydro_text(state, hyd, a2l), hydro.ih_text(ih, a2l)) == ("", "")


def test_partial_hydrogenation_cites_the_hydro_locants():
    _, _, state, a2l = _state("c1ccc2c(c1)C1CCC2C2CCCCC12")
    ih, hyd = hydro.choose_indicated_hydrogen(state, a2l)
    assert hydro.hydro_text(state, hyd, a2l) == "1,2,3,4,4a,9,9a,10-octahydro"


def test_indicated_hydrogen_must_leave_a_mancude_parent():
    # (:24641): indicated hydrogen is cited "consistent with the maximum number
    # of noncumulative double bonds"; a bridge from the fusion atom 4a to C2 leaves nine
    # atoms that can carry a double bond, a path 4-3-2-1-8a-8-7-6-5, so only 4, 2, 8a, 7
    # and 5 can carry the one indicated hydrogen. '2,5-dihydro-1H-' and '1,2-dihydro-5H-'
    # also read back to the dev row's key; 1H is not a position of the mancude parent
    # (OPSIN 2.9.0 cannot assign the double bonds of '1H-2,4a-methanonaphthalene') and
    # 5H loses to 2H:16880, lowest locants to indicated hydrogen).
    _, _, state, a2l = _state("CC1(C)C2=CC=CC(C)(C)C23C=CC1C3")
    assert all(a2l[a] != 1 for s in state.ih_sets for a in s)
    ih, hyd = hydro.choose_indicated_hydrogen(state, a2l)
    assert (hydro.ih_text(ih, a2l), hydro.hydro_text(state, hyd, a2l)) == ("2H", "1,5-dihydro")


def test_mancude_parent_with_a_fusion_atom_bridge_has_its_indicated_hydrogen_at_c2():
    # (:24641): "cited, if possible, at the lowest nonfusion peripheral atom"
    _, _, state, a2l = _state("C=1C2C=CC3(C=CC=CC13)CC2")
    ih, hyd = hydro.choose_indicated_hydrogen(state, a2l)
    assert (hydro.ih_text(ih, a2l), hydro.hydro_text(state, hyd, a2l)) == ("2H", "")


def test_indicated_hydrogen_prefers_a_nonfusion_atom_over_a_lower_fusion_locant():
    # (:24641) "cited, if posible, at the lowest nonfusion peripheral atom",
    # also with hydro prefixes::24685 '3a,5-dihydro-4H-indene (PIN) (not
    # 4,5-dihydro-3aH-indene)'. With 4a, 5, 6 saturated on a 2,8a-bridged naphthalene the
    # indicated hydrogen goes to 6, not to the lower fusion locant 4a.
    _, _, state, a2l = _state("C1=C2C=CC3CCC=CC13C2")
    ih, hyd = hydro.choose_indicated_hydrogen(state, a2l)
    assert (hydro.ih_text(ih, a2l), hydro.hydro_text(state, hyd, a2l)) == ("6H", "4a,5-dihydro")
    # a nonfusion atom that also has the lowest locant
    _, _, state, a2l = _state("C1=C2CCC3C=CC=CC13C2")
    ih, _ = hydro.choose_indicated_hydrogen(state, a2l)
    assert hydro.ih_text(ih, a2l) == "3H"
