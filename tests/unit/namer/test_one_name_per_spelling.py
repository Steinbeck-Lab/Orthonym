"""One molecule, one name, whatever order its SMILES writes its atoms in (the contributor guide core principle 4).

The engine named the atoms in the order a string wrote them, so a choice between two parts the
rules rank alike fell to that order. Two causes, one fix each:

* the parent of two rings of one seniority, a suffix or a prefix form, the order of the locants:
  the input scope of a public call (``namer._lone_pair_input_enter``) now names RDKit's canonical
  SMILES of the molecule the string draws (``input_mol.canonical_spelling``), proven to draw the
  same molecule, from every public entry;
* two names that differ only in stereodescriptors tied in ``alpha_sort_key``, leaving the
  descriptors out of the order, and a ``sorted`` of them kept the input order at the sites that
  sort a Counter or a dict of names (esters, amides, anhydrides, ureas, silanes, gem-ring prefixes).
  ``alpha_sort_key_total`` (assembly/naming_utils.py) adds the CIP rank (j), ``****``
  'R' before 'S') and the name as given, and those sites use it. A site whose items carry locants cites
  'R' before 'S' whatever the locants (``****``, ``:45465``): ``alpha_sort_key_with_locants``.

The spellings below are rows that gave two names before the change (a dev split, the best-effort
tier; the phosphate at both tiers); the names themselves are not pinned here, only that every
spelling gets one.
"""
import itertools

import pytest
from rdkit import Chem

from orthonym import Orthonym, classify_limit, name_compound, name_with_tree, namer
from orthonym import input_mol
from orthonym.assembly.naming_utils import (alpha_sort_key, alpha_sort_key_total,
                                            alpha_sort_key_with_locants, cip_descriptor_rank_key)
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.unit


def _engine(tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier))


def _tier_kwargs(tier):
    flags = _emit_tier_flags(tier)
    return {k: flags[k] for k in ("general_fallback", "general_fallback_unverified",
                                  "allow_aromatic_general", "full_coverage")}


def _spellings(smiles, extra=(), n=4):
    """The input as given, the explicit spellings, and ``n`` seeded RDKit random spellings."""
    mol = Chem.MolFromSmiles(smiles)
    random = [s for s in Chem.MolToRandomSmilesVect(mol, 3 * n, randomSeed=11) if s != smiles][:n]
    out = list(dict.fromkeys([smiles, *extra, *random]))
    assert len(out) >= 3 and all(Chem.MolFromSmiles(s) is not None for s in out)
    assert any(Chem.MolToSmiles(Chem.MolFromSmiles(s)) != s for s in out), "no re-spelling"
    return out


# (class, SMILES as the corpus gives it, a spelling that gave another name before the change)
PARENT_TIES = [
    ("ring parent of two tied rings", "COc1ccc(CCC(O)c2c(O)cc(OC)cc2OC)cc1",
     "c1(c(C(O)CCc2ccc(cc2)OC)c(O)cc(OC)c1)OC"),
    ("prefix on one ring or the other", "CCOc1ccc(S(=O)(=O)NCc2ccccn2)c2ccccc12",
     "c1c(c2ccccc2c(OCC)c1)S(=O)(=O)NCc1ccccn1"),
    ("ring assembly, which ring is the parent", "COc1cc(-c2ccc(O)cc2)c(OC)c(O)c1-c1ccc(O)c(O)c1",
     "O(C)c1c(cc(c(c1O)-c1cc(O)c(O)cc1)OC)-c1ccc(cc1)O"),
    ("two rings joined by a propane chain", "COc1ccc(CCCc2ccc(O)c(OC)c2)c(O)c1",
     "c1c(ccc(O)c1OC)CCCc1ccc(OC)cc1O"),
    ("ester as a prefix or as an ester", "Cc1cc(C)c(C2=C(OC(=O)CC(C)(C)C)C3(CCCC3)OC2=O)c(C)c1",
     "c1c(C)cc(c(C2=C(C3(CCCC3)OC2=O)OC(CC(C)(C)C)=O)c1C)C"),
]


@pytest.mark.parametrize("label, smiles, other", PARENT_TIES, ids=[p[0] for p in PARENT_TIES])
def test_every_spelling_gets_one_best_effort_name(label, smiles, other):
    engine = _engine("best-effort")
    names = {s: engine.name(s) for s in _spellings(smiles, (other,))}
    assert all(names.values()), names
    assert len(set(names.values())) == 1, names


#: the phosphate that gave '2,3-dihydroxypropyl (2R)-...' for 9 spellings and the reverse for 4
PHOSPHATE = "O=P([O-])(OCC(O)CO)OC[C@H](O)CO"
PHOSPHATE_REVERSED_BEFORE = ["[C@@H](COP(=O)([O-])OCC(O)CO)(O)CO",
                             "P(OC[C@H](O)CO)(=O)([O-])OCC(O)CO",
                             "OC[C@H](COP([O-])(OCC(O)CO)=O)O",
                             "[C@H](COP([O-])(OCC(CO)O)=O)(CO)O"]
#: no descriptor before a descriptor, 'R' before 'S' (the rank of ``alpha_sort_key``,
PHOSPHATE_NAME = "2,3-dihydroxypropyl (2R)-2,3-dihydroxypropyl phosphate"


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_phosphate_has_one_name_in_every_spelling(tier):
    engine = _engine(tier)
    names = {s: engine.name(s) for s in _spellings(PHOSPHATE, PHOSPHATE_REVERSED_BEFORE, n=6)}
    assert set(names.values()) == {PHOSPHATE_NAME}, names


def test_two_owners_that_differ_only_in_configuration_are_cited_in_one_order():
    from orthonym.rules.phosphorus import _assemble_p_owner_text
    owners = ["(2R)-2,3-dihydroxypropyl", "2,3-dihydroxypropyl"]
    assert (_assemble_p_owner_text(owners) == _assemble_p_owner_text(owners[::-1])
            == "2,3-dihydroxypropyl (2R)-2,3-dihydroxypropyl")


# ---- every public entry opens the input scope before any scope of it sees the string ----------

ENTRY_ROWS = [PARENT_TIES[0][1:], PARENT_TIES[1][1:]]


def _entry_name(entry, smiles):
    if entry == "name":
        return _engine("best-effort").name(smiles)
    if entry == "name_compound":
        return name_compound(smiles, **_tier_kwargs("best-effort"))
    if entry == "name_compound_confidence":
        return name_compound(smiles, include_confidence=True, **_tier_kwargs("best-effort"))["name"]
    if entry == "name_tiered":
        return _engine("best-effort").name_tiered(smiles)["name"]
    if entry == "name_with_confidence":
        return _engine("best-effort").name_with_confidence(smiles)["name"]
    if entry == "name_with_tree":
        return _engine("best-effort").name_with_tree(smiles).name
    raise AssertionError(entry)


ENTRIES = ["name", "name_compound", "name_compound_confidence", "name_tiered",
           "name_with_confidence", "name_with_tree"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("smiles, other", ENTRY_ROWS, ids=["propane-tied", "naphthalene"])
def test_every_public_entry_names_both_spellings_alike(entry, smiles, other):
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == smiles != other
    assert _entry_name(entry, smiles) == _entry_name(entry, other)


@pytest.mark.parametrize("entry", ENTRIES)
def test_every_public_entry_canonicalises_the_string_it_is_given(entry, monkeypatch):
    """Spy, validated on two rows: the entry reaches ``canonical_spelling`` with the caller's own
    string and the scope it opens names the canonical string in its place."""
    seen = []
    real = input_mol.canonical_spelling

    def spy(smiles):
        out = real(smiles)
        seen.append((smiles, None if out is None else out[0]))
        return out

    monkeypatch.setattr(input_mol, "canonical_spelling", spy)
    for smiles, other in ENTRY_ROWS:
        seen.clear()
        _entry_name(entry, other)
        assert seen and seen[0][0] == other, (entry, seen)
        assert seen[0][1] == smiles, (entry, seen)
        assert [s for s, _ in seen].count(other) == 1, "a nested call named the string again"


def test_the_module_level_entries_and_classify_limit_open_the_scope(monkeypatch):
    seen = []
    real = input_mol.canonical_spelling
    monkeypatch.setattr(input_mol, "canonical_spelling",
                        lambda s: (seen.append(s), real(s))[1])
    smiles, other = ENTRY_ROWS[0]
    assert name_with_tree(other).name == name_with_tree(smiles).name
    assert other in seen
    seen.clear()
    classify_limit(other)
    assert other in seen


def test_a_mol_is_named_as_its_canonical_smiles_is():
    smiles, other = ENTRY_ROWS[0]
    engine = _engine("best-effort")
    assert engine.name(Chem.MolFromSmiles(other)) == engine.name(smiles)


# ---- the string named is proven to be the molecule the caller drew -----------------------------

def _atom_key(atom):
    return (atom.GetSymbol(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(), atom.GetAtomMapNum())


#: SMILES that are not RDKit's canonical SMILES of their molecule, with what a drawing carries:
#: stereo at a centre, a double bond and a ring, isotopes, charges, explicit hydrogens, radicals,
#: dot-separated components, atom maps, a lone-pair centre, brackets (a drawing with a dative bond or
#: a metal is below: RDKit's writer brackets its neighbours)
CARRIED = [
    "OCC", "C(C)(N)[C@@H](O)C", "C(=C/C)\\C", "C1CC[C@H](C)CC1C", "[Cl-].C[N+](C)(C)C",
    "O=C([O-])C.[Na+]", "C[CH2]", "[CH3:1]O", "[CH3][CH2][OH]",
    "[18OH]C(=O)[13CH3]", "C[N+]([O-])(C)C", "c1ccc[nH]1", "OC(=O)[C@@H](N)Cc1c[nH]c2ccccc12",
    "P(C)(CC)c1ccccc1", "[S@@](C)(=O)c1ccccc1",
]


@pytest.mark.parametrize("raw", CARRIED)
def test_the_canonical_spelling_draws_the_same_molecule(raw):
    stage = input_mol.canonical_spelling(raw)
    assert stage is not None, raw
    key, order = stage
    given, named = Chem.MolFromSmiles(raw), Chem.MolFromSmiles(key)
    assert Chem.MolToSmiles(named) == key != raw
    assert Chem.MolToInchi(named) == Chem.MolToInchi(given)
    assert len(order) == given.GetNumAtoms() == named.GetNumAtoms()
    for k, index in enumerate(order):                    # atom k of the string is atom index of the reading
        assert _atom_key(named.GetAtomWithIdx(k)) == _atom_key(given.GetAtomWithIdx(index))
    back = {index: k for k, index in enumerate(order)}
    assert len(back) == len(order)
    for bond in given.GetBonds():
        twin = named.GetBondBetweenAtoms(back[bond.GetBeginAtomIdx()], back[bond.GetEndAtomIdx()])
        assert twin is not None and twin.GetBondType() == bond.GetBondType()


@pytest.mark.parametrize("raw", ["CCS[Zn]SCC", "[Cl-].[Zn+]c1ccccc1", "N->[Pt](<-N)(Cl)Cl",
                                 "C[Sb](c1ccccc1)CC", "O->[Zn+2]<-O"])
def test_a_canonical_smiles_that_would_bracket_more_atoms_is_not_named(raw):
    """RDKit's writer brackets the atoms that bind a metal or a dative bond ('CC[S][Zn][S]CC'), and a
    bracket atom has no implicit hydrogens: a producer that cuts a bond reads a radical there, and
    the plain spelling's name was lost from the bracketed one (the Al drawing of
    tests/unit/rules/test_cage_lambda_and_sort_key.py). Such a string is named as written."""
    given = Chem.MolFromSmiles(raw)
    key = Chem.MolToSmiles(given)
    assert key != raw
    assert (sum(a.GetNoImplicit() for a in Chem.MolFromSmiles(key).GetAtoms())
            > sum(a.GetNoImplicit() for a in given.GetAtoms()))
    assert input_mol.canonical_spelling(raw) is None


@pytest.mark.parametrize("raw", ["CCO", "c1ccccc1", "CC[C@H](C)O", "", "C(C", "*CC",
                                 "C[C@H](O)CC |&1:1|", "C[C@H](O)CC |$_AV:a;b;c;d$|"])
def test_a_string_that_stays_as_written(raw):
    """Canonical already (no proof is needed), unreadable, empty, or CXSMILES (a plain SMILES holds
    none of its extensions): named as written."""
    assert input_mol.canonical_spelling(raw) is None


def test_a_failed_proof_leaves_the_string_as_written(monkeypatch):
    def refuse(given):
        raise input_mol.InvalidInputError("Invalid molecule: the proof does not hold")

    monkeypatch.setattr(input_mol, "_proven_smiles_and_order", refuse)
    assert input_mol.canonical_spelling("OCC") is None


def test_a_canonical_smiles_that_does_not_read_back_as_itself_is_not_named(monkeypatch):
    monkeypatch.setattr(input_mol, "_proven_smiles_and_order",
                        lambda given: ("C(C)O", list(range(given.GetNumAtoms()))))
    assert input_mol.canonical_spelling("OCC") is None


def test_the_scope_keeps_the_caller_string_as_written_for_the_exit_check():
    raw = "[S@@](C)(=O)c1ccccc1"            # a lone-pair centre written first
    named, opened = namer._lone_pair_input_enter(raw)
    try:
        assert opened and named != raw
        assert namer._lone_pair_as_written(named) == raw
    finally:
        namer._lone_pair_input_exit(opened)
    raw = "OCC"
    named, opened = namer._lone_pair_input_enter(raw)
    try:
        assert named == "CCO" and namer._lone_pair_as_written(named) == raw
        assert namer._LP_INPUT.rewrite[2] == [2, 1, 0]
        assert namer._lone_pair_input_enter(raw) == (raw, False)     # a nested call
    finally:
        namer._lone_pair_input_exit(opened)
    assert getattr(namer._LP_INPUT, "rewrite", None) is None


@pytest.mark.parametrize("raw", ["Clc1ccccc1O", "c1(Cl)c(O)cccc1", "c1ccc(O)c(Cl)c1"])
def test_atom_maps_of_a_result_stay_keyed_to_the_callers_string(raw):
    """2-chlorophenol: the ring atom that bears the OH is locant 1 and the one that bears the Cl
    is locant 2, whichever atom of the caller's string each is (not of the string that is named)."""
    assert Chem.MolToSmiles(Chem.MolFromSmiles(raw)) != raw
    hint = Orthonym().name_with_confidence(raw)["atom_to_locant"]
    mol = Chem.MolFromSmiles(raw)
    bearer = {n.GetSymbol(): a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()
              for n in a.GetNeighbors() if not n.IsInRing()}
    assert hint[bearer["O"]] == 1 and hint[bearer["Cl"]] == 2
    assert sorted(hint.values()) == [1, 2, 3, 4, 5, 6]


# ---- the sort key: first, then the configuration (j),, then the name ---

CONFIG_ORDER = ["2,3-dihydroxypropyl", "(2R)-2,3-dihydroxypropyl", "(2S)-2,3-dihydroxypropyl"]


@pytest.mark.parametrize("names", list(itertools.permutations(CONFIG_ORDER)))
def test_names_that_differ_only_in_configuration_sort_alike_whatever_their_input_order(names):
    assert sorted(names, key=alpha_sort_key_total) == CONFIG_ORDER


def test_z_is_cited_before_e_although_it_is_later_in_the_alphabet():
    names = ["(2E)-but-2-en-1-yl", "(2Z)-but-2-en-1-yl"]
    assert sorted(names, key=alpha_sort_key_total) == names[::-1]
    assert sorted(names[::-1], key=alpha_sort_key_total) == names[::-1]


def test_the_configuration_tier_does_not_outrank_a_letter_or_a_locant():
    #: the letters, then the locants, decide first; the descriptor never does
    assert alpha_sort_key_total("(2S)-bromoethyl") < alpha_sort_key_total("(2R)-chloroethyl")
    assert alpha_sort_key_total("(2S)-1-chloroethyl") < alpha_sort_key_total("(2R)-2-chloroethyl")
    assert cip_descriptor_rank_key("(2R)-2-chloroethyl") < cip_descriptor_rank_key("(2S)-2-chloroethyl")


def test_the_key_stays_a_total_order_on_names_that_differ_only_in_a_descriptor_locant():
    a, b = alpha_sort_key_total("(1R)-1-chloroethyl"), alpha_sort_key_total("(2R)-1-chloroethyl")
    assert a != b and (a < b) != (b < a)


def test_alpha_sort_key_itself_still_ties_what_p14_5_leaves_out():
    """ (``the Blue Book``) leaves stereochemical descriptors out of the alphanumerical order:
    the key is unchanged, so a locant that follows it in a tuple key still decides first."""
    assert alpha_sort_key("(2R)-2,3-dihydroxypropyl") == alpha_sort_key("2,3-dihydroxypropyl")
    e, z = "(E)-C,N-dihydroxycarbonimidoyl", "(Z)-C,N-dihydroxycarbonimidoyl"
    rows = [(z, 5), (e, 2)]
    for order in (rows, rows[::-1]):
        assert [n for n, _ in sorted(order, key=lambda r: (alpha_sort_key(r[0]), r[1]))] == [e, z]


# ---- the sites that sort names by it: one name, the configuration cited as the book orders it ----

#: (class, SMILES, the name). 'R' before 'S',:22603), 'Z' before 'E' (j),:3346); two
#: names that differ only in configuration were cited in the order the atoms were written in.
CITED_BY_CONFIGURATION = [
    ("diester of a diacid", "C[C@H](Cl)OC(=O)CC(=O)O[C@@H](C)Cl",
     "(1R)-1-chloroethyl (1S)-1-chloroethyl propanedioate"),
    ("phosphate triester", "COP(=O)(O[C@H](C)Cl)O[C@@H](C)Cl",
     "(1R)-1-chloroethyl (1S)-1-chloroethyl methyl phosphate"),
    ("sulfate diester", "C/C=C\\OS(=O)(=O)O/C=C/C", "(1Z)-prop-1-en-1-yl (1E)-prop-1-en-1-yl sulfate"),
    ("mixed anhydride", "C/C=C\\C(=O)OC(=O)/C=C/C", "(2Z)-but-2-enoic (2E)-but-2-enoic anhydride"),
    ("N,N-disubstituted amide", "CC(=O)N([C@H](C)Cl)[C@@H](C)Cl",
     "N-[(1R)-1-chloroethyl]-N-[(1S)-1-chloroethyl]acetamide"),
    ("N,N'-disubstituted urea", "C[C@H](Cl)NC(=O)N[C@@H](C)Cl",
     "N-[(1R)-1-chloroethyl]-N'-[(1S)-1-chloroethyl]urea"),
    ("silane", "CC[C@@H](C)[SiH2][C@@H](C)CC", "[(2R)-butan-2-yl][(2S)-butan-2-yl]silane"),
    ("gem-disubstituted ring", "C[C@H](Cl)C1([C@@H](C)Cl)CCCCC1",
     "1-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclohexane"),
    ("without a descriptor first", "CP(=O)(OCC(O)CO)OC[C@H](O)CO",
     "2,3-dihydroxypropyl (2R)-2,3-dihydroxypropyl methylphosphonate"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("label, smiles, expected", CITED_BY_CONFIGURATION,
                         ids=[c[0] for c in CITED_BY_CONFIGURATION])
def test_configuration_decides_the_citation_of_names_that_tie_in_every_spelling(label, smiles, expected, tier):
    engine = _engine(tier)
    names = {s: engine.name(s) for s in _spellings(smiles, n=5)}
    assert set(names.values()) == {expected}, names


# ---- a site whose items carry locants: the configuration comes before the locants of the names -----

def test_the_configuration_comes_before_the_locants_a_site_passes():
    """``### **** Sequence Rule **4a**`` (``the Blue Book``), its example (``:45465``): ``(2R,3s,4S,6R)-
    2,6-dichloro-5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]heptan-4-ol (PIN)`` cites R first at the HIGHER locant
    5; ``****`` (``:22603``, under ``### **** CRITERIA RELATED ONLY TO CONFIGURATION``) has the order of
    'R' and 'S' decide when the names tie in. ``****`` (``:3517``) compares the locants INSIDE the names
    ("the fact that 'N' is lower than '4' is irrelevant", ``:3527``); (``:21791``) is a criterion of the
    numbering, and ``:22601`` a rejected name that applies it to one. The locants a site passes order only
    what the book leaves."""
    s, r = "(2S)-butan-2-yl", "(2R)-butan-2-yl"
    rows = [(r, 3), (s, 1)]
    assert [n for n, _ in sorted(rows, key=lambda x: alpha_sort_key_with_locants(x[0], (x[1],)))] == [r, s]
    assert [n for n, _ in sorted(rows, key=lambda x: alpha_sort_key_total(x[0]))] == [r, s]   # no locant: R first
    same = [(s, 2), (r, 2)]
    assert [n for n, _ in sorted(same, key=lambda x: alpha_sort_key_with_locants(x[0], (x[1],)))] == [r, s]


def test_the_n_substituents_of_a_urea_are_cited_r_first_whatever_their_locants():
    """The tags are given here (N for the S group, N' for the R group): R is cited first, at N' (:45465)."""
    from orthonym.assembly.composer import _build_n_substituted_name
    s, r = "(1S)-1-chloroethyl", "(1R)-1-chloroethyl"
    for tagged in ([("N", s), ("N'", r)], [("N'", r), ("N", s)]):
        assert _build_n_substituted_name(tagged, "urea", sort_key=alpha_sort_key) == (
            "N'-[(1R)-1-chloroethyl]-N-[(1S)-1-chloroethyl]urea")
    for tagged in ([("N", s), ("N", r)], [("N", r), ("N", s)]):
        assert _build_n_substituted_name(tagged, "guanidine", sort_key=alpha_sort_key) == (
            "N-[(1R)-1-chloroethyl]-N-[(1S)-1-chloroethyl]guanidine")


#: (what, SMILES as written, the name). A diamine whose two nitrogens carry a (2R)- and a (2S)-butan-2-yl:
#: on butane-1,3-diamine the numbering is fixed, so the (2R) group stands at N3 and is cited first, before the
#: (2S) group at N1,:45465: R cited first at the higher locant;,:22603); on the symmetric
#: propane-1,3-diamine the numbering is a choice and (j) (:3346) gives the lower locant to R, so N1 is
#: the (2R) side. The numbering of the symmetric chain was the order the atoms were written in.
POLYAMINES = [
    ("numbering fixed by the suffix", "CC[C@@H](C)NC(C)CCN[C@@H](C)CC",
     "N3-[(2R)-butan-2-yl]-N1-[(2S)-butan-2-yl]butane-1,3-diamine"),
    ("symmetric chain, R at the lower locant", "C[C@@H](CC)NCCCN[C@H](C)CC",
     "N1-[(2R)-butan-2-yl]-N3-[(2S)-butan-2-yl]propane-1,3-diamine"),
    ("the same chain written the other way", "CC[C@@H](C)NCCCN[C@@H](C)CC",
     "N1-[(2R)-butan-2-yl]-N3-[(2S)-butan-2-yl]propane-1,3-diamine"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("label, smiles, expected", POLYAMINES, ids=[p[0] for p in POLYAMINES])
def test_polyamine_prefixes_cite_r_first_in_every_spelling(label, smiles, expected, tier):
    engine = _engine(tier)
    names = {s: engine.name(s) for s in _spellings(smiles, n=6)}
    assert set(names.values()) == {expected}, names


#: The same diamine with an isotopic label. NUMBERING (``:3219``): clause (i) (``:3336``) "...lowest
#: locants to the modified atoms or groups considered together as a set... If a choice still remains, the
#: lower locant is given to the nuclide of higher atomic number... of higher mass number", and "isotopic
#: modifications have seniority for low locants over stereodescriptors described in (j)" (``:3346``). So
#: the numbering of the symmetric chain goes to the label first, and to R (j) only where the labels tie
#: (a label in the middle, on both ends, or none). The label was stripped before the skeleton was named, and
#: the (j) tie-break renumbered a chain whose label then took the higher locant ((3-13C), (3,3-2H2)).
LABELLED_POLYAMINES = [
    ("13C at C1, on the (S) side: (i) puts it at 1", "C[C@@H](CC)N[13CH2]CCN[C@H](C)CC",
     "N3-[(2R)-butan-2-yl]-N1-[(2S)-butan-2-yl](1-13C)propane-1,3-diamine"),
    ("13C at C3: (i) puts it at 1, R follows it to N1 only by luck of the structure",
     "C[C@@H](CC)NCC[13CH2]N[C@H](C)CC",
     "N1-[(2R)-butan-2-yl]-N3-[(2S)-butan-2-yl](1-13C)propane-1,3-diamine"),
    ("2H2 at C1: the deuterium path names the unlabelled skeleton", "C[C@@H](CC)NC([2H])([2H])CCN[C@H](C)CC",
     "N3-[(2R)-butan-2-yl]-N1-[(2S)-butan-2-yl](1,1-2H2)propane-1,3-diamine"),
    ("13C in the middle: (i) ties, (j) gives R the lower locant", "C[C@@H](CC)NC[13CH2]CN[C@H](C)CC",
     "N1-[(2R)-butan-2-yl]-N3-[(2S)-butan-2-yl](2-13C)propane-1,3-diamine"),
    ("13C on both ends: (i) ties, (j) gives R the lower locant", "C[C@@H](CC)N[13CH2]C[13CH2]N[C@H](C)CC",
     "N1-[(2R)-butan-2-yl]-N3-[(2S)-butan-2-yl](1,3-13C2)propane-1,3-diamine"),
    ("13C and 14C on the ends: the higher mass number takes the lower locant", "C[C@@H](CC)N[13CH2]C[14CH2]N[C@H](C)CC",
     "N1-[(2R)-butan-2-yl]-N3-[(2S)-butan-2-yl](3-13C,1-14C)propane-1,3-diamine"),
    ("a labelled substituent group takes the lower locant", "C[C@@H](CC)NCCCN[C@H]([13CH3])CC",
     "N1-[(2R)-(1-13C)butan-2-yl]-N3-[(2S)-butan-2-yl]propane-1,3-diamine"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("label, smiles, expected", LABELLED_POLYAMINES,
                         ids=[p[0][:40] for p in LABELLED_POLYAMINES])
def test_an_isotopic_label_numbers_the_chain_before_the_configuration(label, smiles, expected, tier):
    engine = _engine(tier)
    names = {s: engine.name(s) for s in _spellings(smiles, n=6)}
    assert set(names.values()) == {expected}, names
