"""The peptide route offers its names; a chiral diamide with substituted nitrogens is named (lane/l17 item 6).

Lead 8 of `internal notes` asked whether ``routing.dispatch_table._handle_peptide`` returns the peptide name and
blocks a route that names the molecule better. Measured at lane/l17 (5cd9d81c3a, ``Orthonym(_seed_excluded_dispatch_classes=
{PEPTIDE})`` against the default call): at the PIN tier no name or tier changes on 210 druglike rows, 255 ChEBI peptide-like
rows or the 9 gold rows that reach the handler, and the PIN-sample time is the same (257.7 s against 259.0 s). The test
below pins that, so a route that begins to block a PIN shows.

L5 of the same file: the two 14-residue witnesses of ``test_peptide_capped_termini.py`` ended at different names. The handler
prefers the substitutive name of its sub-namer because it is the PIN (decision A); a systematic name that breaks a rule of the
PIN spelling is offered after the peptide name. The Leu witness got an amido name whose parent is the leucinamide chain, one
amide group, while the asparagine residue is a chain with two "SENIORITY ORDER FOR PARENT STRUCTURES",,
the Blue Book: "The senior parent structure has the maximum number of substituents corresponding to the principal
characteristic group (suffix)..."); the Phe witness got none. Both keep the peptide name now.

Why the Phe witness got none: ``rules.diamide_chain_prefixes.chain_carbon_prefixes`` declined a structure with a stereo layer,
and ``amides.name_chain_diamide`` then left a chiral butanediamide with a substituted amide nitrogen to the general path,
which names its far carbonyl as a CH2 ('(3S)-3-acetamido-4-(methylamino)butanamide', a different molecule that the round trip
withdraws). The diamide namer cites the descriptors of its chain now, the Blue Book: at the front of the complete
name) when an amide nitrogen is substituted.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from orthonym.routing.dispatch_table import StoutClass, _breaks_pin_spelling
from tests.support.default_tier import strict_path_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_TIERS = ["pin", "best-effort"]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# a chiral butanediamide (or an Asn peptide with a C-terminal amide) with a substituted amide nitrogen
_DIAMIDES = [
    ("CC(=O)N[C@@H](CC(N)=O)C(=O)NC", "(2S)-2-acetamido-N1-methylbutanediamide"),
    ("CC(=O)N[C@H](CC(N)=O)C(=O)NC", "(2R)-2-acetamido-N1-methylbutanediamide"),
    ("NC(=O)CC(NC(C)=O)C(=O)N[C@@H](C)C(N)=O", "2-acetamido-N1-[(2S)-1-amino-1-oxopropan-2-yl]butanediamide"),
    ("N[C@@H](C)C(=O)N[C@@H](CC(N)=O)C(=O)NCC", "(2S)-2-[(2S)-2-aminopropanamido]-N1-ethylbutanediamide"),
    ("N[C@@H](C)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](C)C(N)=O",
     "(2S)-N1-[(2S)-1-amino-1-oxopropan-2-yl]-2-[(2S)-2-aminopropanamido]butanediamide"),
    ("CC(C)(CO)NC(=O)[C@H](CC=C)CC(=O)NC1(CCCC1)CO",
     "(2R)-N4-[1-(hydroxymethyl)cyclopentyl]-N1-(1-hydroxy-2-methylpropan-2-yl)-2-(prop-2-en-1-yl)butanediamide"),
    ("CCNC(=O)[C@H](C)CC(=O)NC", "(2R)-N1-ethyl-N4,2-dimethylbutanediamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _DIAMIDES)
def test_a_chiral_diamide_with_substituted_nitrogens_is_named_on_its_chain(smiles, name, tier):
    """ (the Blue Book) cites the N-substituents with the locant N on the diamide parent
    ('N1-methyl...butanediamide'); (:44639) puts the descriptors of the parent at the front of the name."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the diamides the general path names
_UNCHANGED = [
    ("CC(=O)NC(CC(N)=O)C(=O)NC", "2-acetamido-N1-methylbutanediamide", "pin_verified"),
    ("NC(=O)[C@H](O)[C@@H](O)C(N)=O", "(2R,3R)-2,3-dihydroxybutanediamide", "pin_verified"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name,label", _UNCHANGED)
def test_the_diamides_the_general_path_names_keep_their_names(smiles, name, label, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, label), row


# the two numberings of a symmetric chain tie on every criterion the diamide namer reads before the descriptors of the chain
_SYMMETRIC = [
    # C2 symmetric: both numberings read '(2R,3R)', so the name is the same in either
    ("CNC(=O)[C@H](O)[C@@H](O)C(=O)NC", "(2R,3R)-2,3-dihydroxy-N1,N4-dimethylbutanediamide"),
    ("CNC(=O)[C@@H](O)[C@H](O)C(=O)NC", "(2S,3S)-2,3-dihydroxy-N1,N4-dimethylbutanediamide"),
    ("CNC(=O)[C@H](C)[C@@H](C)C(=O)NC", "(2R,3R)-N1,N4,2,3-tetramethylbutanediamide"),
    # meso: '(2R,3S)' or '(2S,3R)' by the numbering, and the R takes the lower locant (j)), whatever the
    # direction of the SMILES
    ("CNC(=O)[C@H](O)[C@H](O)C(=O)NC", "(2R,3S)-2,3-dihydroxy-N1,N4-dimethylbutanediamide"),
    ("CNC(=O)[C@@H](O)[C@@H](O)C(=O)NC", "(2R,3S)-2,3-dihydroxy-N1,N4-dimethylbutanediamide"),
    ("CNC(=O)[C@H](C)[C@H](C)C(=O)NC", "(2R,3S)-N1,N4,2,3-tetramethylbutanediamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _SYMMETRIC)
def test_a_symmetric_chiral_diamide_is_named_with_the_descriptors_by_criterion_j(smiles, name, tier):
    """ NUMBERING (the Blue Book), criterion (g) (:3307) "lowest locants for the substituent cited first as a
    prefix in the name" and (j) (:3346) "the lower locant is assigned to CIP stereodescriptors Z, R, M, and r
    (pseudoasymmetry) that are preferred to E, S, P, and s". 2,3-Dihydroxy-N1,N4-dimethylbutanediamide ties on (f) and (g) in
    both directions; the C2 symmetric diastereomer reads the same either way, and the meso one is numbered so that the R takes
    the lower locant. The PIN tier ships both, full-key read back."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the two near-identical 14-residue witnesses of test_peptide_capped_termini.py
_ILE_TO_PHE_NH2 = (
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)"
    "N[C@H](C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)"
    "N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C(N)=O)C(C)C)[C@@H](C)CC"
)
_ILE_TO_LEU_NH2 = (
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)"
    "N[C@H](C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)"
    "N[C@@H](CCCCN)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(N)=O)[C@@H](C)CC"
)
# what the sub-namer of the handler built for the Leu witness at 5cd9d81c3a: a name that reads back to the input
_LEU_AMIDO_NAME = (
    "(2S)-2-[(2S)-2-{(2S)-6-amino-2-[(2S)-6-amino-2-{(2S)-2-[(2S)-2-{(2S)-2-[(2S)-2-{(2S,3S)-2-[(2S)-2-{(2S)-6-amino-2-"
    "[(2S)-2-{(2S)-4-amino-2-[(2S,3S)-2-amino-3-methylpentanamido]-4-oxobutanamido}-4-methylpentanamido]hexanamido}"
    "propanamido]-3-methylpentanamido}propanamido]propanamido}-3-phenylpropanamido]propanamido}hexanamido]hexanamido}"
    "-4-methylpentanamido]-4-methylpentanamide"
)


def test_the_two_witnesses_keep_the_peptide_name():
    """The handler prefers a substitutive name when it is the PIN; the Leu witness's amido name breaks, so the peptide
    name stands for both witnesses (the Phe one never had a substitutive name)."""
    for smiles, end in ((_ILE_TO_PHE_NH2, "phenylalaninamide"), (_ILE_TO_LEU_NH2, "leucinamide")):
        row = strict_path_row(smiles)
        assert row["name"].startswith("isoleucylasparaginyl") and row["name"].endswith(end), row
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
        assert row["producer"] == "outer:peptide", row
        assert_full_rt(row["name"], smiles)


def test_the_amido_name_of_the_leu_witness_breaks_p_44_1_1_and_reads_back_to_the_input():
    """The name that was displaced is right for the molecule and is no PIN: the check finds in it (a parent of one
    amide group, a chain of the structure with two), and it is withheld by the offer, never lost."""
    mol = Chem.MolFromSmiles(_ILE_TO_LEU_NH2)
    assert_full_rt(_LEU_AMIDO_NAME, _ILE_TO_LEU_NH2)
    assert _breaks_pin_spelling(_LEU_AMIDO_NAME, mol)
    from orthonym.validation.pin_spelling import check_pin_spelling
    assert {f.rule for f in check_pin_spelling(mol, _LEU_AMIDO_NAME)} == {"P-44.1.1"}


def test_a_clean_substitutive_name_is_still_preferred():
    """The offer changes nothing for a name that is the PIN: no failure, so the sub-namer's name is returned."""
    smiles = "N[C@@H](C)C(=O)N[C@@H](C)C(=O)O"
    name = "(2S)-2-[(2S)-2-aminopropanamido]propanoic acid"
    assert not _breaks_pin_spelling(name, Chem.MolFromSmiles(smiles))
    for tier in _TIERS:
        row = _row(smiles, tier)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), row


# a residue whose retained acyl name carries a ring that needs an indicated hydrogen ('histidyl', 'tryptophyl'), in a tripeptide
# whose arginine the systematic name keeps as the parent: the same hybrid as the one of phenylalanine
_HYBRIDS = [
    ("NC(N)=NCCC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)O",
     "(2S)-5-[(diaminomethylidene)amino]-2-(histidylaspartylamino)pentanoic acid"),
    ("NC(N)=NCCC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",
     "(2S)-5-[(diaminomethylidene)amino]-2-(tryptophylaspartylamino)pentanoic acid"),
    ("NC(N)=NCCC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",
     "(2S)-5-[(diaminomethylidene)amino]-2-(phenylalanylaspartylamino)pentanoic acid"),
]


@pytest.mark.parametrize("smiles,name", _HYBRIDS)
def test_a_retained_residue_prefix_is_no_indicated_hydrogen_failure_of_the_substitutive_name(smiles, name):
    """The check read 'histidyl' and 'tryptophyl' as an imidazole and an indole named without their indicated hydrogen
    , so the substitutive name was offered after the peptide name. The residue names carry the hydrogen of their
    ring, Table 10.4;: the name's only failure is the mark every name of the chapter has, and
    it ships at the best-effort tier as before, like the phenylalanine one (no PIN is identified for it,."""
    assert not _breaks_pin_spelling(name, Chem.MolFromSmiles(smiles))
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), row
    assert_full_rt(row["name"], smiles)
    assert _row(smiles, "pin")["tier"] == "abstain"


def test_a_name_that_breaks_a_spelling_rule_is_not_lost_without_a_peptide_name(monkeypatch):
    """A spelling-flawed substitutive name ships when the molecule has no peptide name to prefer: the offer orders names, it
    withdraws none."""
    monkeypatch.setattr("orthonym.rules.peptides.name_peptide", lambda mol: None)
    row = strict_path_row(_ILE_TO_LEU_NH2)
    assert row["name"] == _LEU_AMIDO_NAME, row
    assert row["is_pin"] is False, row


def _handler_record(monkeypatch, smiles):
    """Name ``smiles`` at the best-effort tier with the peptide handler traced: what the handler returned for the molecule
    itself and the provenance and the stored candidate it left at the return."""
    import dataclasses
    from orthonym.assembly.coverage_scoring import _confidence_store
    from orthonym.metrics.provenance import get_provenance
    from orthonym.routing import dispatch_table as dt
    seen = []
    entry = dt.DISPATCH_TABLE[StoutClass.PEPTIDE]

    def spy(mol, *args, **kwargs):
        name = entry.handler(mol, *args, **kwargs)
        if mol.GetNumHeavyAtoms() == Chem.MolFromSmiles(smiles).GetNumHeavyAtoms():
            seen.append((name, get_provenance(), getattr(_confidence_store, "last_candidate", None)))
        return name

    monkeypatch.setitem(dt.DISPATCH_TABLE, StoutClass.PEPTIDE, dataclasses.replace(entry, handler=spy))
    _row(smiles, "best-effort")
    return seen[-1]


def test_a_flawed_name_that_ships_keeps_what_its_attempt_recorded(monkeypatch):
    """The name that ships for want of a peptide name is the one the attempt built: the gate outcome it was cleared by
    and the candidate whose locant map the stereo backstop reads are put back with it."""
    monkeypatch.setattr("orthonym.rules.peptides.name_peptide", lambda mol: None)
    name, prov, candidate = _handler_record(monkeypatch, _ILE_TO_LEU_NH2)
    assert name == _LEU_AMIDO_NAME
    assert prov["gate_outcome_name"] == name, prov["gate_outcome_name"]
    assert candidate is not None


def test_an_offered_name_leaves_no_record_behind_the_peptide_name(monkeypatch):
    """The reverse: the name that gives way to the peptide name leaves neither its gate outcome nor its candidate."""
    name, prov, candidate = _handler_record(monkeypatch, _ILE_TO_LEU_NH2)
    assert name.startswith("isoleucylasparaginyl"), name
    assert prov["gate_outcome_name"] != _LEU_AMIDO_NAME
    assert candidate is None


# the 9 gold rows that reach the handler (benchmarks/the gold set: gold_pins.json, packs/characteristic_groups.json)
_GOLD_PEPTIDE_ROWS = [
    ("N[C@@H](C(C)C)C(=O)N[C@@H](CC1=CC=C(C=C1)O)C(=O)N[C@@H]([C@@H](C)CC)C(=O)O",
     "(2S,3S)-2-{(2S)-2-[(2S)-2-amino-3-methylbutanamido]-3-(4-hydroxyphenyl)propanamido}-3-methylpentanoic acid"),
    ("NC(=O)CC[C@H](N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O",
     "(2S)-2-{(2S)-6-amino-2-[(2S)-2,5-diamino-5-oxopentanamido]hexanamido}-5-(carbamimidoylamino)pentanoic acid"),
    ("C[C@@H](C(=O)N[C@@H](C)C(=O)O)N", "(2S)-2-[(2S)-2-aminopropanamido]propanoic acid"),
    ("NCC(=O)N[C@@H](C)C(=O)O", "(2S)-2-(2-aminoacetamido)propanoic acid"),
    ("C[C@H](C(=O)NCC(=O)O)N", "[(2R)-2-aminopropanamido]acetic acid"),
    ("NC(=O)NCCC[C@H](N)C(=O)O", "(2S)-2-amino-5-(carbamoylamino)pentanoic acid"),
    ("CNC(=O)NC(CC(C)=O)C(=O)O", "2-[(methylcarbamoyl)amino]-4-oxopentanoic acid"),
    ("CCCCC(=O)NC(CCC(=O)O)C(=O)O", "2-pentanamidopentanedioic acid"),
    ("CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)C(=O)O", "(2S)-3-hydroxy-2-octadecanamidopropanoic acid"),
]


@pytest.mark.parametrize("smiles,pin", _GOLD_PEPTIDE_ROWS)
def test_the_peptide_route_blocks_no_pin_at_the_pin_tier(smiles, pin):
    """Lead 8, measured: at the PIN tier the route and the cascade without it give the gold PIN alike."""
    from orthonym import Orthonym
    with_route = Orthonym(style="pin").name_tiered(smiles)
    without = Orthonym(style="pin", _seed_excluded_dispatch_classes=frozenset({StoutClass.PEPTIDE})).name_tiered(smiles)
    assert (with_route["name"], with_route["tier"]) == (pin, "pin_verified"), with_route
    assert (without["name"], without["tier"]) == (with_route["name"], with_route["tier"]), without
