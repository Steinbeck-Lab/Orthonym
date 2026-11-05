"""LO3: the bridged fused PIN builder's name is offered when the older fused-heterocycle namer's
name is known not to be the PIN.

``fused_rings.name_fused_heterocycle`` names a fused ring system from two fusion components and
spells a skeleton that has a retained name by its descriptor ('benzo[d]pyrimidine' for
quinazoline, 'benzo[b]pyrazine' for quinoxaline, '4H-cyclohexa[b]pyran'), or a ketone on a
saturated position by a hydro prefix; it records the spelling as a non-PIN fragment, so the name
is labelled below the PIN and the default tier declines the molecule. The builder
(``bridged_fused_pin.build_fused``) names the same molecules from the PIN parents, but it was
reached only when the older namer declined, so it never ran for them.

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* NAMES OF HETEROCYCLIC PARENT RING COMPONENTS, (:11505): "Retained names for
  heterocycles (also called trivial names) with the maximum number of noncumulative double bonds
  (called mancude ring systems) used for parent components and as attached components are given
  in Table 2.8"; Table 2.8 (:11511) gives "(9) quinazoline (PIN)" (:11553). The descriptor name
  'benzo[d]pyrimidine' is therefore not the name of the parent.
* SUBSTITUENT GROUPS MODIFIED BY THE PREFIXES 'HYDRO' OR 'DEHYDRO' (:16870),
  (:16872): the hydro prefixes "are cited immediately at the front of the name of the parent
  hydride", and the numbering gives "lowest locants to fixed numbering of a polycyclic ring
  system, as in naphthalene, quinoline, etc." (:16876): '1,4-dihydroquinazolin-2-amine'.
* (:24221): "Retained fusion names are used for the fully unsaturated compounds (see
  ; they are the preferred IUPAC names. Preferred IUPAC names for the partially saturated and
  fully saturated compounds are formed by using 'hydro' prefixes." Polycyclic mancude
  compounds (:17024),:17026: "Total hydrogenation is indicated by the appropriate multiplicative
  prefixes indicating the total number of hydrogen atoms attached, but locants are omitted": the
  saturated bicycles take 'octahydroindolizine', 'hexahydro-1H-pyrrolizine' (Table 2.8 (21)
  "the PIN is 1H-pyrrolizine",:11628), 'octahydro-2H-quinolizine' (Table 2.8 (15) "the PIN is
  4H-quinolizine",:11582), not 'indolizidine', 'pyrrolizidine', 'quinolizidine' (0 hits in the
  book). ASSUMED (no book example): '2H' is the lowest indicated-hydrogen locant that exists for
  quinolizine (1H would leave C4 without a double-bond partner). '(3aR,7aS)-octahydro-1H-indole
  (PIN)' (:49612) is the book's own total-hydrogenation spelling.
* 'Added indicated hydrogen' (:24687): "'Added indicated hydrogen' is cited in parentheses
  after the locant of the structural feature to which it refers. This method is preferred over
  the use of nondetachable hydro prefixes for preferred IUPAC names." (:24689):
  '2H-1,4-benzoxazin-3(4H)-one', not '3,4-dihydro-2H-1,4-benzoxazin-3-one'. The 1-benzopyran
  parent is '2H-1-benzopyran (PIN)' (:19453), '3,4-dihydro-2H-1-benzopyran (PIN)' (:24673).
* Localized double bonds (:14595),:14597: "If it is necessary to identify isomers that
  differ only by virtue of the location of localized double bonds, this differentiation is
  indicated by the use of the Greek letter Δ" ('1,6-dimethyl-Δ1-heptalene (PIN)'): the standard
  InChI does not see the location of the localized double bonds of a non-aromatic mancude ring
  system, so the read-back of an offered name must be the SAME DRAWING, not only the same InChI
  (`test_a_bond_shift_partner_is_not_offered`).

The offer is made only when (1) the older name carries a recorded non-PIN part, (2) the builder's
name carries none, (3) OPSIN reads it back to the full InChI and (4) the whole ring system is
covered. A molecule the builder cannot name better keeps the older name.
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import (
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# smiles, the PIN (both tiers, pin_verified), the older namer's name (right molecule, not the PIN)
PIN_ROWS = [
    # the item's rows: the fused guanidine / amidine of the fu4 record (internal notes)
    ("NC1=NCc2ccccc2N1", "1,4-dihydroquinazolin-2-amine",
     "1H,4H-benzo[d]pyrimidin-2-amine"),
    ("NC1=NCc2c(ccc(Cl)c2Cl)N1", "5,6-dichloro-1,4-dihydroquinazolin-2-amine",
     "5,6-dichloro-1H,4H-benzo[d]pyrimidin-2-amine"),
    # the same descriptor spelling on other quinazoline and quinoxaline hydro forms
    ("C1=NCc2ccccc2N1", "1,4-dihydroquinazoline", "3H,4H-benzo[d]pyrimidine"),
    ("C1=NCc2cc(Cl)ccc2N1", "6-chloro-1,4-dihydroquinazoline", "6-chloro-1H,4H-benzo[d]pyrimidine"),
    ("C1=CNc2cc(C)ccc2N1", "6-methyl-1,4-dihydroquinoxaline", "6-methyl-1H,4H-benzo[b]pyrazine"),
    ("Nc1nc2NCCNc2c(=O)[nH]1", "2-amino-5,6,7,8-tetrahydropteridin-4(3H)-one", "tetrahydropterin"),
    ("CN1C=CCc2ccccc21", "1-methyl-1,4-dihydroquinoline", "1-methyl-4H-benzo[b]pyridine"),
    ("Cc1ncc2c(n1)CSC2", "2-methyl-5,7-dihydrothieno[3,4-d]pyrimidine",
     "2-methyl-5H,7H-thieno[3,4-d]pyrimidine"),
    # a ketone on a saturated position: added hydrogen, not a hydro prefix
    ("O=C1Nc2ccccc2OC1", "2H-1,4-benzoxazin-3(4H)-one", "3,4-dihydro-2H-1,4-benzoxazin-3-one"),
    ("O=C1COc2ccc(Cl)cc2N1", "6-chloro-2H-1,4-benzoxazin-3(4H)-one",
     "6-chloro-3,4-dihydro-2H-1,4-benzoxazin-3-one"),
    ("C[C@@H]1OCc2c(O)cccc2C1=O", "(3S)-8-hydroxy-3-methyl-1H-2-benzopyran-4(3H)-one",
     "(3S)-8-hydroxy-3-methyl-3,4-dihydro-1H-2-benzopyran-4-one"),
    # a skeleton with a retained name spelled by its descriptor, or a trivial name that is no PIN
    ("O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12", "5,7-dihydroxy-2-(4-hydroxyphenyl)-4H-1-benzopyran-4-one",
     "apigenin"),
    ("O=C(O)[C@@H]1Cc2cc(O)c(O)cc2C=N1", "(3S)-6,7-dihydroxy-3,4-dihydroisoquinoline-3-carboxylic acid",
     "(3S)-6,7-dihydroxy-4H-benzo[c]pyridine-3-carboxylic acid"),
    ("C1CCN2CCCC2C1", "octahydroindolizine", "indolizidine"),
    ("C1CCC2CCCCN2C1", "octahydro-2H-quinolizine", "quinolizidine"),
]


@pytest.mark.parametrize("smiles,pin,older", PIN_ROWS)
def test_builder_name_replaces_a_known_non_pin_name(smiles, pin, older):
    """The builder's name ships at both tiers as the PIN; the older name was right but is no PIN."""
    assert name_is_rt_exact(older, smiles), f"the older name no longer reads back: {older!r}"
    assert_pin_at_both_tiers(smiles, pin)
    assert_not_pin_labelled(smiles, older)


def test_item_rows_best_effort_producer_and_default_tier():
    """The reproducers of the record: best-effort was systematic_verified, default abstained."""
    for smiles, pin in (("NC1=NCc2ccccc2N1", "1,4-dihydroquinazolin-2-amine"),
                        ("NC1=NCc2c(ccc(Cl)c2Cl)N1", "5,6-dichloro-1,4-dihydroquinazolin-2-amine")):
        d = name_default(smiles)
        assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (pin, "pin_verified", True), d
        b = name_breadth(smiles)
        assert (b.get("name"), b.get("tier"), b.get("producer")) == (
            pin, "pin_verified", "inner:tier_a_ring"), b


# The builder names this too, but its name carries a non-PIN part (the 'phosphonooxy' prefix is in the
# non-PIN vocabulary, ``rules/pin_vocabulary.non_pin_vocabulary``), so an offer that is labelled below the
# PIN as well gains no tier: the older name stays. (LO4a: the two acyl rows that stood here, the pteridine row
# '2-amino-6-(2-hydroxy-1-oxopropyl)-...' and its pyrazino[2,3-d]pyrimidine twin, are PINs now that the
# acyl group is named as an acyl group, tests/unit/assembly/test_lo4a_acyl_prefix.py.)
STILL_NOT_PIN_ROWS = [
    # change-asserted-value (lane/l17 fix a performance pass; it already failed at c302d83df4, from item 12): the old name was
    # '6-chloro-2-[5-chloro-2-(phosphonooxy)phenyl]quinazolin-4(1H)-one' (systematic_verified). (the Blue Book)
    # puts "9 Esters" (:18182) above the ketone-like '-one' ring group (class 16,:18189), so the phosphate ester is the parent
    #,:35916 'methyl dihydrogen phosphate (PIN)',:35940) and the quinazolinone a prefix of its organyl group; the
    # name is in PIN form with the hydro-prefix spelling of a drawn 1H tautomer that no strict producer certifies, so it is
    # pin_unverified (a breadth producer built it) and the default tier declines it. Read back below by ``name_is_rt_exact``;
    # mutation: with ``seniority._OXOACID_ESTERS_PROMOTED`` empty the old name returns.
    ("O=c1nc(-c2cc(Cl)ccc2OP(=O)(O)O)[nH]c2ccc(Cl)cc12",
     "4-chloro-2-(6-chloro-4-oxo-1,4-dihydroquinazolin-2-yl)phenyl dihydrogen phosphate"),
]


@pytest.mark.parametrize("smiles,name", STILL_NOT_PIN_ROWS)
def test_offer_that_is_also_non_pin_changes_nothing(smiles, name):
    assert name_is_rt_exact(name, smiles), name
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (name, "pin_unverified"), b
    assert name_default(smiles).get("tier") != "pin_verified"


# A PIN already: the older namer's name carries no non-PIN part, so the builder is not asked.
@pytest.mark.parametrize("smiles,pin", [
    ("c1ccc2ncccc2c1", "quinoline"),
    ("Cc1ccc2ncccc2c1", "6-methylquinoline"),
    ("O=C1CCc2ccccc2N1", "3,4-dihydroquinolin-2(1H)-one"),
])
def test_pin_names_are_unchanged(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# A bond-shift partner is a different drawing with the same standard InChI: the builder names the
# partner ('1-methylheptalene' for the drawing whose own name is '5-methylheptalene'), OPSIN reads
# it back to the InChI of the input, and the offer would label it a PIN:14595,
#:14597: such isomers are told apart by the Δ descriptor). The read-back is the same drawing.
BOND_SHIFT_ROWS = [
    ("CC1=C2C=CC=CC=C2C=CC=C1", "5-methylheptalene", "1-methylheptalene"),
    ("CC1=C2C=C3C=CC=C3C=C2C=C1", "3-methyl-s-indacene", "1-methyl-s-indacene"),
]


@pytest.mark.parametrize("smiles,older,partner", BOND_SHIFT_ROWS)
def test_a_bond_shift_partner_is_not_offered(smiles, older, partner):
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (older, "systematic_verified"), b
    assert name_default(smiles).get("tier") != "pin_verified"
    assert name_is_rt_exact(older, smiles)


@pytest.mark.parametrize("smiles,older,partner", BOND_SHIFT_ROWS)
def test_read_back_is_the_same_drawing_not_the_same_inchi(smiles, older, partner):
    from rdkit.Chem.inchi import MolToInchi

    from orthonym.rules.fused_pin_offer import _reads_back
    from orthonym.validation.opsin_roundtrip import opsin_parse
    mol = Chem.MolFromSmiles(smiles)
    back = Chem.MolFromSmiles(opsin_parse(partner))
    assert MolToInchi(mol) == MolToInchi(back), "the partner shares the standard InChI of the drawing"
    assert Chem.MolToSmiles(mol) != Chem.MolToSmiles(back)
    assert _reads_back(mol, partner) is False
    assert _reads_back(mol, older) is True


# ---------------------------------------------------------------------------------------------
# The decision function, on the real namers.
# ---------------------------------------------------------------------------------------------
def _covers_everything(mol, atom_to_locant):
    return True


@pytest.fixture
def fresh_provenance():
    from orthonym.metrics.provenance import clear_provenance
    clear_provenance()
    yield
    clear_provenance()


def _offer(smiles, covers=_covers_everything):
    from orthonym.rules.fused_pin_offer import name_fused_with_pin_offer
    return name_fused_with_pin_offer(Chem.MolFromSmiles(smiles), covers)


def test_offer_returns_the_builders_tuple(fresh_provenance):
    res = _offer("NC1=NCc2ccccc2N1")
    assert res is not None and res[0] == "1,4-dihydroquinazolin-2-amine", res
    name, ring_atoms, atom_to_locant, subs_included = res
    assert subs_included is True and set(atom_to_locant) <= set(ring_atoms), res


def test_older_name_kept_when_it_is_not_known_non_pin(fresh_provenance, monkeypatch):
    """No recorded non-PIN part: the builder is not even asked."""
    import orthonym.rules.bridged_fused as bf
    calls = []
    monkeypatch.setattr(bf, "name_fused_carbocycle_pin",
                        lambda mol: calls.append(1) or ("never", set(), {}, True))
    res = _offer("Cc1ccc2ncccc2c1")
    assert res is not None and res[0] == "6-methylquinoline", res
    assert calls == []


def test_older_name_kept_when_the_read_back_fails(fresh_provenance, monkeypatch):
    """The builder's name is not shipped unless OPSIN reads it back to the molecule (or when no
    OPSIN can be run): the older name stays, whatever the builder says."""
    import orthonym.rules.fused_pin_offer as fo
    monkeypatch.setattr(fo, "_reads_back", lambda mol, name: False)
    res = _offer("NC1=NCc2ccccc2N1")
    assert res is not None and res[0] == "1H,4H-benzo[d]pyrimidin-2-amine", res


def test_older_name_kept_when_the_ring_system_is_not_covered(fresh_provenance):
    res = _offer("NC1=NCc2ccccc2N1", covers=lambda mol, atom_to_locant: False)
    assert res is not None and res[0] == "1H,4H-benzo[d]pyrimidin-2-amine", res


def test_older_name_kept_when_the_offer_is_also_non_pin(fresh_provenance):
    res = _offer("O=c1nc(-c2cc(Cl)ccc2OP(=O)(O)O)[nH]c2ccc(Cl)cc12")
    assert res is not None
    assert res[0] == "6-chloro-2-[5-chloro-2-(phosphonooxy)phenyl]quinazolin-4-one", res


def test_older_name_kept_when_the_builder_declines(fresh_provenance):
    """The older namer cites a group senior to the ring ('hydroxymethyl' beside the ring, a
    recorded non-PIN part) and the builder has no name for this molecule (it only spells a
    plain -NH2 or ketone suffix): the older name stays and the builder is asked in vain."""
    from orthonym.rules.bridged_fused import name_fused_carbocycle_pin
    smiles = "OCc1ccc2ccccc2n1"
    assert name_fused_carbocycle_pin(Chem.MolFromSmiles(smiles)) is None
    res = _offer(smiles)
    assert res is not None and res[0] == "2-(hydroxymethyl)quinoline", res


def test_no_older_name_returns_none(fresh_provenance):
    """The offer never invents a name where the older namer declined: the caller's later
    branches (including the builder itself) run as before."""
    assert _offer("C1CCCCC1") is None


def test_known_non_pin_is_name_scoped(fresh_provenance):
    from orthonym.metrics.provenance import get_provenance, record_non_pin_fragment
    from orthonym.rules.fused_pin_offer import known_non_pin
    record_non_pin_fragment("benzo[d]pyrimidin")
    prov = get_provenance()
    assert known_non_pin(prov, "1H,4H-benzo[d]pyrimidin-2-amine")
    assert not known_non_pin(prov, "1,4-dihydroquinazolin-2-amine")
    # the whole-call flags lower every name of the call equally: they cannot separate two candidates
    assert not known_non_pin(dict(prov, general_ring_prefix=True, pin_promotion_rerun=True),
                             "1,4-dihydroquinazolin-2-amine")
