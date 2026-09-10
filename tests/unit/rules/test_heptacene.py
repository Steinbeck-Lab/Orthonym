"""Heptacene (7-fused-ring linear acene, — new POLYCYCLIC_DATA entry.

Was fail-closed at PIN tier ('unknown organic compound' -- the general fused-ring
engine has no catalog entry for a 7-ring linear acene) prior to this fix. OPSIN
2.9.0 parses 'heptacene' (verified: `echo heptacene | java -jar
opsin-cli-2.9.0-jar-with-dependencies.jar -o extendedsmi` succeeds), so the name
round-trips; the missing piece was purely the Orthonym-side catalog entry.

The `iupac_numbering` map (canonical-SMILES atom index -> IUPAC locant) is
AUTHORITATIVE: derived from OPSIN's own `heptacene -o extendedsmi` locants
(`$_AV:1;2;3;4;4a;5;5a;6;6a;7;7a;8;8a;9;9a;10;11;12;13;13a;14;14a;15;15a;16;16a;
17;17a;18;18a`), mapped onto this project's RDKit-canonical SMILES via
`Mol.GetSubstructMatch` (confirmed the same molecule by identical InChI between
the OPSIN-ordered parse and the RDKit-canonical parse).

Per-position round-trip verification (18/18 peripheral integer locants, done in
a scratchpad script before this entry was added -- see
`.superpowers/sdd/heptacene-fix-report.md`): a mono-methyl derivative was built
at every one of heptacene's 18 cataloged peripheral atoms, named via Orthonym,
and round-tripped through OPSIN to the exact input InChIKey. All 18 positions
matched (0-wrong). Several physical positions (3,4,8,9,...) come back renamed to
a LOWER locant than their raw map value -- that is heptacene's own D2h/C2h
molecular symmetry (an unsubstituted linear acene's terminal-ring alpha/beta
positions and its meso edge-CH positions each form a symmetry-equivalent class,
exactly as for naphthalene's 1=4=5=8 / 2=3=6=7), and
`get_polycyclic_iupac_locants`'s existing automorphism-based lowest-locant
selector (already load-bearing for hexacene/pentacene) correctly collapses each
physical position to its class's lowest member per / (a).
This test only asserts positions that are ALREADY their class's lowest member
(1, 2, 6, 7), so the naive "locant in name" check is a correct oracle here.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = pytest.mark.opsin_gate

HEPTACENE_SMILES = "c1ccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc2c1"
HEXACENE_SMILES = "c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1"
PENTACENE_SMILES = "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1"
ANTHRACENE_SMILES = "c1ccc2cc3ccccc3cc2c1"


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smi, name):
    """True iff `name` round-trips through OPSIN to the exact input InChIKey."""
    opsin_smi = opsin_parse(name)
    if not opsin_smi:
        return False
    m_in = Chem.MolFromSmiles(smi)
    m_out = Chem.MolFromSmiles(opsin_smi)
    if m_in is None or m_out is None:
        return False
    return inchi.MolToInchiKey(m_in) == inchi.MolToInchiKey(m_out)


def test_heptacene_bare(namer):
    name = namer.name(HEPTACENE_SMILES)
    assert name == "heptacene", name
    assert _full_rt(HEPTACENE_SMILES, name), name


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # locant 1: terminal-ring "alpha" position -- already its symmetry
        # class's lowest member (verified: raw-position round trip gave
        # '1-methylheptacene' unchanged).
        ("Cc1cccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc12", "1-methylheptacene"),
        # locant 2: terminal-ring "beta" position, likewise its own class's
        # lowest member.
        ("Cc1ccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc2c1", "2-methylheptacene"),
        # locant 6: one of the meso edge-CH classes (bromo, to also exercise a
        # non-methyl substituent), lowest member of its class.
        ("Brc1c2cc3ccccc3cc2cc2cc3cc4cc5ccccc5cc4cc3cc12", "6-bromoheptacene"),
        # locant 7: the unique central-ring meso position (only 2-fold
        # degenerate under heptacene's symmetry, since it sits on the
        # molecule's own left-right mirror axis) -- lowest member of its class.
        ("Brc1c2cc3cc4ccccc4cc3cc2cc2cc3cc4ccccc4cc3cc12", "7-bromoheptacene"),
    ],
)
def test_substituted_heptacene_locant_and_rt(namer, smiles, expected):
    name = namer.name(smiles)
    assert name == expected, (smiles, name)
    assert _full_rt(smiles, name), (smiles, name)


def test_shorter_acenes_not_shadowed(namer):
    """A longer acene's SMARTS/catalog entry must not shadow or mis-match a
    shorter one (or vice versa) -- num_atoms + the exact-coverage check in
    identify_polycyclic prevent it; this asserts the emitted behaviour.

    (tetracene is deliberately excluded here: its substituted-form emission
    already used the 'naphthacene' spelling before this change -- a
    pre-existing, unrelated quirk out of this fix's scope -- so it is not a
    useful regression oracle for THIS change.)
    """
    assert namer.name(ANTHRACENE_SMILES) == "anthracene"
    assert namer.name(PENTACENE_SMILES) == "pentacene"
    assert namer.name(HEXACENE_SMILES) == "hexacene"
    assert namer.name(HEPTACENE_SMILES) == "heptacene"

    # And the substituted forms stay routed to their own (not each other's) core.
    assert namer.name("Cc1cccc2cc3ccccc3cc12") == "1-methylanthracene"
    assert namer.name("Cc1cccc2cc3cc4cc5ccccc5cc4cc3cc12") == "1-methylpentacene"
    assert namer.name(
        "Cc1cccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc12"
    ) == "1-methylhexacene"
    assert namer.name(
        "Cc1cccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc12"
    ) == "1-methylheptacene"


def test_num_atoms_disambiguates_hexacene_vs_heptacene():
    """Direct data-level check: the two catalog entries have different
    num_atoms, so a 26-atom hexacene core can never satisfy a 30-atom
    heptacene SMARTS match (and vice versa) -- identify_polycyclic's
    'core_atoms == pah_data[\"num_atoms\"]' exact-coverage gate depends on
    this staying true."""
    from orthonym.data.polycyclic_data import POLYCYCLIC_DATA

    assert POLYCYCLIC_DATA["hexacene"]["num_atoms"] == 26
    assert POLYCYCLIC_DATA["heptacene"]["num_atoms"] == 30
    assert POLYCYCLIC_DATA["hexacene"]["num_rings"] == 6
    assert POLYCYCLIC_DATA["heptacene"]["num_rings"] == 7

    hexacene_mol = Chem.MolFromSmiles(HEXACENE_SMILES)
    heptacene_pattern = Chem.MolFromSmarts(
        POLYCYCLIC_DATA["heptacene"]["smarts"]
    )
    assert not hexacene_mol.HasSubstructMatch(heptacene_pattern)

    heptacene_mol = Chem.MolFromSmiles(HEPTACENE_SMILES)
    hexacene_pattern = Chem.MolFromSmarts(POLYCYCLIC_DATA["hexacene"]["smarts"])
    # A 26-atom hexacene pattern CAN embed inside a larger uncapped fragment in
    # principle, but identify_polycyclic's core_atoms==num_atoms coverage gate
    # (num_atoms 26 != heptacene's own atom count 30, and the largest-first
    # scan tries heptacene's own 30-atom pattern first and returns on that
    # exact match) means heptacene is never misnamed via the hexacene entry --
    # asserted end-to-end above (test_shorter_acenes_not_shadowed).
    assert heptacene_mol.HasSubstructMatch(hexacene_pattern), (
        "sanity: heptacene should contain a hexacene-shaped substructure "
        "(this is fine -- identify_polycyclic's largest-first + exact-"
        "coverage logic is what prevents mis-routing, not SMARTS non-overlap)"
    )
