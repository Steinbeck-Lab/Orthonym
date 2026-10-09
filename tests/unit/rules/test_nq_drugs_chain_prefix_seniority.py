"""A chain parent of a cyclic molecule is checked against to with its ring
prefixes counted, located and ordered; a name built on a chain that is not the senior one is
kept at every tier but never labelled pin_verified (``assembly.candidate_pool``,
``perception.chains.chain_parent_prefix_seniority``).

- (the Blue Book) "the maximum number of substituents cited as prefixes"; the
  chain parent of example (4),:21624, counts its ring substituent.
- (:21698) the lower locant set for prefixes.
- (:21791) "the lower locant or set of locants for substituents cited as prefixes...
  in their order of citation in the name"; '3-bromo-2-(2-bromo-1-hydroxyethyl)-4-hydroxybutanoic
  acid (PIN) [not 4-bromo-2-(1-bromo-2-hydroxyethyl)-3-hydroxybutanoic acid;... '3,2,4' in the
  PIN is lower than '4,2,3']' (:22108).

The chain selector excludes ring atoms from chain membership and, before this check, also from
the prefix count, the prefix locants and the citation-order tie-break, so a chain that ends at a
ring lost to one that did not.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.perception.chains import find_principal_chain
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

TIERS = ["pin", "valid", "complete", "best-effort"]

#: (SMILES, the name the producer builds, the PIN by; the first row is the dev2000 row
#: the lane's prefixes first certified, the second main's own certified row (the known positive)
#:: equal counts and locant sets; citation-order locants (2,3) in the PIN, (3,2) built
P4523_ROWS = [
    ("N#CC(CO)Cc1cc([N+](=O)[O-])ccc1N1CCCC1",
     "3-hydroxy-2-{[5-nitro-2-(pyrrolidin-1-yl)phenyl]methyl}propanenitrile",
     "2-(hydroxymethyl)-3-[5-nitro-2-(pyrrolidin-1-yl)phenyl]propanenitrile"),
    ("N#CC(CO)Cc1cccc([N+](=O)[O-])c1",
     "3-hydroxy-2-[(3-nitrophenyl)methyl]propanenitrile",
     "2-(hydroxymethyl)-3-(3-nitrophenyl)propanenitrile"),
    ("N#CC(CO)Cc1ccc(cc1)-c1ccccn1",
     "3-hydroxy-2-{[4-(pyridin-2-yl)phenyl]methyl}propanenitrile",
     "2-(hydroxymethyl)-3-[4-(pyridin-2-yl)phenyl]propanenitrile"),
    ("OC(=O)C(CO)Cc1ccc(cc1)-n1ccnc1",
     "3-hydroxy-2-{[4-(1H-imidazol-1-yl)phenyl]methyl}propanoic acid",
     "2-(hydroxymethyl)-3-[4-(1H-imidazol-1-yl)phenyl]propanoic acid"),
    ("OC(=O)C(CO)Cc1ccc(cc1)N1CCCC1",
     "3-hydroxy-2-{[4-(pyrrolidin-1-yl)phenyl]methyl}propanoic acid",
     "2-(hydroxymethyl)-3-[4-(pyrrolidin-1-yl)phenyl]propanoic acid"),
    ("CC(=O)C(CO)Cc1ccccc1N1CCCC1",
     "4-hydroxy-3-{[2-(pyrrolidin-1-yl)phenyl]methyl}butan-2-one",
     "3-(hydroxymethyl)-4-[2-(pyrrolidin-1-yl)phenyl]butan-2-one"),
    ("N#CC(CO)Cc1ccccc1N1CCCC1",
     "3-hydroxy-2-{[2-(pyrrolidin-1-yl)phenyl]methyl}propanenitrile",
     "2-(hydroxymethyl)-3-[2-(pyrrolidin-1-yl)phenyl]propanenitrile"),
    ("OC(=O)C(CCl)Cc1ccc(cc1)-c1ccccn1",
     "3-chloro-2-{[4-(pyridin-2-yl)phenyl]methyl}propanoic acid",
     "2-(chloromethyl)-3-[4-(pyridin-2-yl)phenyl]propanoic acid"),
    # a principal ketone: (3,4) in citation order against (4,3), the '=O' being no prefix
    ("CC(=O)C(CCl)Cc1ccc(F)cc1",
     "4-chloro-3-[(4-fluorophenyl)methyl]butan-2-one",
     "3-(chloromethyl)-4-(4-fluorophenyl)butan-2-one"),
]

#:: two prefixes (methyl, phenyl) on the PIN's chain, one (benzyl) on the built one;
#: one row per producer that builds these names (the chain catch-all, amide, ester)
P4521_ROWS = [
    ("OC(=O)C(C)Cc1ccccc1", "2-benzylpropanoic acid", "2-methyl-3-phenylpropanoic acid"),
    ("NC(=O)C(C)Cc1ccccc1", "2-benzylpropanamide", "2-methyl-3-phenylpropanamide"),
    ("COC(=O)C(C)Cc1ccccc1", "methyl 2-benzylpropanoate", "methyl 2-methyl-3-phenylpropanoate"),
    ("CNC(=O)C(C)Cc1ccccc1", "2-benzyl-N-methylpropanamide", "N,2-dimethyl-3-phenylpropanamide"),
]

#: names that are the PIN and keep pin_verified at every tier
PIN_ROWS = [
    # the control of the finding:, (2,3) here against (3,2) for
    # 3-(4-chlorophenyl)-2-(hydroxymethyl)propanoic acid
    ("OC(=O)C(CO)Cc1ccc(Cl)cc1", "2-[(4-chlorophenyl)methyl]-3-hydroxypropanoic acid"),
    # the -O- connective: (2,3) against (3,2)
    ("N#CC(CO)COc1ccccn1", "2-(hydroxymethyl)-3-[(pyridin-2-yl)oxy]propanenitrile"),
    # equal counts and locants in citation order (2,3) both ways; (:22234), 'benzyl'
    # before 'chloromethyl'
    ("OC(=O)C(CCl)Cc1ccccc1", "2-benzyl-3-chloropropanoic acid"),
    # the same with a principal ketone: its '=O' is the suffix, not an 'oxo' prefix of the
    # key ('benzyl' before 'chloromethyl', as in the acid)
    ("CC(=O)C(CCl)Cc1ccccc1", "3-benzyl-4-chlorobutan-2-one"),
    ("CC(=O)C(CBr)Cc1ccccc1", "3-benzyl-4-bromobutan-2-one"),
    ("CC(=O)C(CF)Cc1ccccc1", "3-benzyl-4-fluorobutan-2-one"),
    # a principal alcohol: 'hydroxy' is not a prefix either
    ("OCC(CCl)Cc1ccccc1", "2-benzyl-3-chloropropan-1-ol"),
    # a longer chain is senior before is reached
    ("OC(=O)C(CC)Cc1ccccc1", "2-benzylbutanoic acid"),
    # two dev2000 rows whose chains end at either methyl of an isopropyl group: chains a
    # symmetry of the molecule maps onto each other are one chain
    ("CC(C)CC(=O)[C@@H](O)Cc1c[nH]c2c(O)cccc12",
     "(2S)-2-hydroxy-1-(7-hydroxy-1H-indol-3-yl)-5-methylhexan-3-one"),
]

#: a row whose name is main's own, not asserted as a PIN here: the chain check must leave its
#: name and label as they are with the check off (main spells a hyphen between 'methyl' and
#: 'isoquinolin', which the book does not: 'isoquinolin-7-yl' is joined to its prefixes,
#: the Blue Book; the spelling is main's, handed to its owner)
CHAIN_CHECK_NEUTRAL_ROWS = [
    "Cc1cc2ccc(C(=O)[C@H](O)C(C)C)c(O)c2cn1",
]


def _tier_row(tier, smiles, namer=None):
    if namer is None:
        namer = (Orthonym(style="pin") if tier == "pin"
                 else Orthonym(style="pin", **_emit_tier_flags(tier)))
    return namer.name_tiered(smiles)


@pytest.mark.parametrize("smiles,built,pin", P4523_ROWS + P4521_ROWS)
def test_the_pin_by_p45_2_reads_back(smiles, built, pin):
    assert name_is_rt_exact(pin, smiles), pin
    assert name_is_rt_exact(built, smiles), built


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("smiles,built,pin", P4523_ROWS[:2])
def test_a_chain_that_is_not_the_senior_one_is_never_certified(tier, smiles, built, pin):
    row = _tier_row(tier, smiles)
    assert row.get("tier") != "pin_verified", row
    if tier == "pin":
        return
    # the name stays below the PIN, read back
    assert (row["name"], row["tier"]) == (built, "systematic_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,built,pin", P4523_ROWS[2:] + P4521_ROWS)
def test_the_class_is_never_certified(tier, smiles, built, pin):
    row = _tier_row(tier, smiles)
    assert row.get("tier") != "pin_verified", row
    if tier == "best-effort":
        assert (row["name"], row["tier"]) == (built, "systematic_verified"), row
        assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_the_senior_chain_keeps_its_pin(tier, smiles, pin):
    row = _tier_row(tier, smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


def test_a_stereodescriptor_does_not_hide_the_name():
    smiles = "N#C[C@@H](CO)Cc1cccc([N+](=O)[O-])c1"
    assert _tier_row("pin", smiles).get("tier") != "pin_verified"
    row = _tier_row("best-effort", smiles)
    assert (row["name"], row["tier"]) == (
        "(2S)-3-hydroxy-2-[(3-nitrophenyl)methyl]propanenitrile", "systematic_verified"), row


def test_the_label_is_what_lowers_the_name(monkeypatch):
    # known positive: without the label the producer's name is certified, as on main (a SMILES
    # used by no other test here, a fresh engine)
    from orthonym.assembly import candidate_pool
    smiles = "N#CC(CO)Cc1ccc([N+](=O)[O-])cc1"
    built = "3-hydroxy-2-[(4-nitrophenyl)methyl]propanenitrile"
    row = _tier_row("pin", smiles)
    assert row.get("tier") != "pin_verified", row
    monkeypatch.setattr(candidate_pool, "_label_chain_parent_prefix_seniority",
                        lambda name, features: None)
    row = _tier_row("pin", smiles)
    assert (row["name"], row["tier"]) == (built, "pin_verified"), row


def _chain_args(smiles):
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group
    mol = Chem.MolFromSmiles(smiles)
    fgs = detect_functional_groups(mol)
    ring_atoms = {a for ring in mol.GetRingInfo().AtomRings() for a in ring}
    return mol, fgs, get_principal_group(mol, fgs)[0], ring_atoms


def test_the_selector_counts_ring_prefixes_only_when_asked():
    # atoms: N0 C1 C2 C3 O4 C5 c6...; C3 bears the OH, C5 the ring
    mol, fgs, pg, ring_atoms = _chain_args("N#CC(CO)Cc1cccc([N+](=O)[O-])c1")
    assert set(find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms)) == {1, 2, 3}
    assert set(find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms,
                                    ring_prefixes=True)) == {1, 2, 5}
    # the control: (4-chlorophenyl)methyl before hydroxy keeps the CH2OH chain
    mol, fgs, pg, ring_atoms = _chain_args("OC(=O)C(CO)Cc1ccc(Cl)cc1")
    assert set(find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms,
                                    ring_prefixes=True)) == {1, 3, 4}


def test_an_unnamed_branch_in_a_tie_is_undecided(monkeypatch):
    from orthonym.assembly import substituent_enumerator
    mol, fgs, pg, ring_atoms = _chain_args("N#CC(CO)Cc1cccc([N+](=O)[O-])c1")
    monkeypatch.setattr(substituent_enumerator, "name_substituent_for_ordering",
                        lambda mol, frag, att: None)
    assert find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms,
                                ring_prefixes=True) is None


def test_the_check_leaves_no_provenance_record(monkeypatch):
    # the comparison names branches speculatively; a non-PIN record made there (here the
    # peptide producer's, for a phenylalanyl branch) must not lower the label of the name
    # the real naming builds: with and without the check the dev2000 peptide row gets the
    # same name and label
    from orthonym.assembly import candidate_pool
    smiles = "NC(=O)CC(NC(=O)C(Cc1ccccc1)NC(=O)C(Cc1ccccc1)NC(=O)C(N)CCCN=C(N)N)C(=O)O"
    with_check = _tier_row("pin", smiles)
    monkeypatch.setattr(candidate_pool, "_label_chain_parent_prefix_seniority",
                        lambda name, features: None)
    without_check = _tier_row("pin", smiles)
    assert (with_check["name"], with_check["tier"]) == (
        without_check["name"], without_check["tier"]), (with_check, without_check)


def test_isolated_provenance_drops_the_records_and_log_entries_of_its_body():
    from orthonym.metrics import provenance
    provenance.clear_provenance()
    provenance.record_non_pin_fragment("kept")
    log = provenance.push_non_pin_log()
    try:
        with provenance.isolated_provenance():
            provenance.record_non_pin_fragment("dropped")
            provenance.record_non_pin_label("dropped-label")
            provenance.record_uncertified_pin_name("dropped-name")
            provenance.record_source("somewhere")
        prov = provenance.get_provenance()
        assert prov["non_pin_fragments"] == ("kept",)
        assert prov["non_pin_labels"] == ()
        assert prov["uncertified_pin_names"] == ()
        assert prov["source"] is None
    finally:
        provenance.pop_non_pin_log()
        provenance.clear_provenance()
    # the record call inside was taken out of the log; the read after the block stays
    assert "dropped" not in log
    assert log == [provenance.PROVENANCE_READ]


def test_the_p45_5_letters_of_the_cited_prefixes():
    # (the Blue Book): letters in the order they appear, multiplied prefixes as
    # cited; 'bromo' is earlier than 'dibromo' (:22245), '4-(1,2-difluoropropyl)-5,6-dinitro'
    # earlier than '4-(1,2-dinitropropyl)-5,6-difluoro' (example (4))
    from orthonym.perception.chains import _cited_prefix_letters
    assert _cited_prefix_letters(["bromo", "chloro"]) == "bromochloro"
    assert _cited_prefix_letters(["bromo", "bromo"]) == "dibromo"
    assert _cited_prefix_letters(["bromo", "chloro"]) < _cited_prefix_letters(["bromo", "bromo"])
    assert (_cited_prefix_letters(["(1,2-difluoropropyl)", "nitro", "nitro"])
            < _cited_prefix_letters(["(1,2-dinitropropyl)", "fluoro", "fluoro"]))
    # italic (upper-case) letters are left out
    assert _cited_prefix_letters(["(1H-indol-3-yl)methyl"]) == "indolylmethyl"


def test_two_different_chains_that_still_tie_are_undecided():
    # protoporphyrin IX: the two propanoic acid chains differ only in the locants of the ring
    # prefix, which the comparison does not read, so it does not claim either is senior; the
    # name keeps main's label (pin_unverified), not 'not the PIN'
    smiles = ("C=CC1=C(C)c2cc3[nH]c(cc4nc(cc5[nH]c(cc1n2)c(C)c5CCC(=O)O)C(CCC(=O)O)=C4C)"
              "c(C)c3C=C")
    mol, fgs, pg, ring_atoms = _chain_args(smiles)
    assert find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms,
                                ring_prefixes=True) is None
    row = _tier_row("best-effort", smiles)
    assert row["tier"] == "pin_unverified", row


@pytest.mark.parametrize("smiles", CHAIN_CHECK_NEUTRAL_ROWS)
def test_the_check_leaves_a_name_that_it_does_not_decide_as_it_is(monkeypatch, smiles):
    from orthonym.assembly import candidate_pool
    for tier in TIERS:
        with_check = _tier_row(tier, smiles)
        monkeypatch.setattr(candidate_pool, "_label_chain_parent_prefix_seniority",
                            lambda name, features: None)
        without_check = _tier_row(tier, smiles)
        monkeypatch.undo()
        assert (with_check["name"], with_check["tier"]) == (
            without_check["name"], without_check["tier"]), (tier, with_check, without_check)


def test_the_principal_group_is_the_suffix_not_a_prefix_of_the_key():
    # (the Blue Book) orders "substituents cited as prefixes"; the ketone's '=O'
    # is the suffix. Atoms: C0 C1(=O2) C3(C4 Cl5) C6 c7..: the chain through the ring-bearing
    # CH2 is the one with the benzyl branch ('benzyl' before 'chloromethyl',:22234),
    # and the chain through the CH2Cl is the other; 'oxo' on the key moved (3,4,2) of the
    # built chain behind (3,2,4)
    mol, fgs, pg, ring_atoms = _chain_args("CC(=O)C(CCl)Cc1ccccc1")
    assert find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms,
                                ring_prefixes=True) == [0, 1, 3, 4]
    # a constructed row: SCC(N)(Cc1ccccc1)C(=O)C(N)CN has the PIN '2,4,5-triamino-1-phenyl-2-
    # (sulfanylmethyl)pentan-3-one' (citation-order locants (2,4,5,1,2) against (2,4,5,2,1)):
    # the chain ends at the CH2 that bears the ring (atom 4), not at the CH2SH carbon (atom 1)
    mol, fgs, pg, ring_atoms = _chain_args("SCC(N)(Cc1ccccc1)C(=O)C(N)CN")
    chain = find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms, ring_prefixes=True)
    assert chain[-1] == 4 and chain[:3] == [15, 13, 11], chain


@pytest.mark.parametrize("smiles", [
    # the two methyl ends of an isopropyl group: chains a symmetry of the molecule maps onto
    # each other
    "CC(C)CC(=O)[C@@H](O)Cc1c[nH]c2c(O)cccc12",
    "Cc1cc2ccc(C(=O)[C@H](O)C(C)C)c(O)c2cn1",
])
def test_chains_that_a_symmetry_maps_onto_each_other_are_decided_without_naming(
        monkeypatch, smiles):
    # the comparison reads only the length and the symmetry key of the chain it returns, so
    # the prefixes (whole branches, large for a peptide) are not named for such a tie
    from orthonym.assembly import substituent_enumerator
    calls = []
    orig = substituent_enumerator.name_substituent_for_ordering
    monkeypatch.setattr(substituent_enumerator, "name_substituent_for_ordering",
                        lambda *a, **k: calls.append(a) or orig(*a, **k))
    mol, fgs, pg, ring_atoms = _chain_args(smiles)
    chain = find_principal_chain(mol, fgs, pg, exclude_atoms=ring_atoms, ring_prefixes=True)
    assert chain is not None and not calls, calls
