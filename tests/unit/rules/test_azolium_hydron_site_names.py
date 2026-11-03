"""LO9: an azolium cation whose ring carries a characteristic group is named from either drawing, so
the rule that ranks the names of one species can rank them.

``Nc1c[nH+]cn1C`` and ``Nc1c[nH]c[n+]1C`` are one species (the formal +1 on either ring nitrogen, one
standard InChIKey). The drawing with the charge on the NH nitrogen had no name (the producers of
``ions`` decline a ring that carries an amino, carboxy or hydroxy group as a principal group), so
``Orthonym._name_with_form_choice`` had one name to choose and shipped the name of the other drawing,
'4-amino-3-methyl-1H-imidazol-3-ium', labelled ``pin_verified``. That is the numbering the PIN tier
does not give a plain azolium ('1,5-dimethyl-1H-imidazol-3-ium', never '3,4-dimethyl-...'): the Blue
Book numbers the cation as the neutral compound, (the Blue Book) "This is consistent
with the choice of lowest locants for corresponding neutral compounds (see ", and
"NUMBERING" (:3219) ranks (b) the indicated hydrogen (:3246), (c) the suffixes (:3256) and then (f)
"detachable alphabetized prefixes, all considered together in a series of increasing numerical
order" (:3301). With the indicated hydrogen at 1 and the 'ium' at 3 in both numberings, the set of the
prefix locants decides: {1,5} ('5-amino-1-methyl-') is lower than {3,4} ('4-amino-3-methyl-'). The
cation is the added hydron of (:41366) "A cation derived formally by adding one or more
hydrons to any position of a neutral parent hydride", '1H-imidazol-3-ium (PIN)' (:41396), with the
substituents replacing hydrogens, so the nitrogen that carries the substituent is the position of
the indicated hydrogen.

Decision LO9 (`internal notes`): the label of the name a
species gets does not depend on whether every drawing could be named, but it does depend on no unnamed
drawing outranking the name that ships; for this class one did, and the producer now names it. Every
name is read back by a FRESH OPSIN call and must give the input's fixed-H InChI.
"""
import pytest

from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: (the two drawings of one species, the PIN the PIN tier builds for both)
PINS = [
    ("Nc1c[nH+]cn1C", "Nc1c[nH]c[n+]1C", "5-amino-1-methyl-1H-imidazol-3-ium"),
    ("Cn1cc(N)[nH+]c1", "C[n+]1cc(N)[nH]c1", "4-amino-1-methyl-1H-imidazol-3-ium"),
    ("O=C(O)c1c[nH+]cn1C", "O=C(O)c1c[nH]c[n+]1C", "5-carboxy-1-methyl-1H-imidazol-3-ium"),
    ("Cn1cc[nH+]c1N", "C[n+]1cc[nH]c1N", "2-amino-1-methyl-1H-imidazol-3-ium"),
    ("Nc1c(C)[nH+]cn1C", "Nc1c(C)[nH]c[n+]1C", "5-amino-1,4-dimethyl-1H-imidazol-3-ium"),
]
#: the N-N azoliums: only the drawing with the charge on the NH nitrogen. The name of the other drawing
#: ('Nc1cc[nH][n+]1C') puts the hydron of the PIN numbering on the other atom than the drawing's, which the
#: protonation-site check of ``charged_router._cation_name_rt_ok`` does not accept for a N-N pair, so that
#: drawing keeps the name it had (recorded in ``prov.md``, LO9, "Left").
NH_PLUS_DRAWING_ONLY = [
    ("Nc1cc[nH+]n1C", "5-amino-1-methyl-1H-pyrazol-2-ium"),
    ("Oc1cc[nH+]n1C", "5-hydroxy-1-methyl-1H-pyrazol-2-ium"),
]
#: the rows of the policy: the unnamed drawing is a prefix-only cation, the name that ships keeps its label
UNNAMED_RIVAL_KEEPS_THE_LABEL = [
    ("NC1=[NH+]C(C(=O)[O-])CS1", "2-amino-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
    ("NC1=[NH+][C@@H](C(=O)[O-])CS1", "(4S)-2-amino-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
    ("NC1=[NH+][C@H](C(=O)[O-])CS1", "(4R)-2-amino-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
    ("C[N+]1=C(N)SCC1C(=O)[O-]", "2-amino-3-methyl-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
    ("NC1=[N+](C)C(C(=O)[O-])CS1", "2-amino-3-methyl-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
]
TIERS = ["pin", "best-effort"]


def _engine(tier):
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    return (Orthonym(style="pin") if tier == "pin"
            else Orthonym(style="pin", **_emit_tier_flags(tier)))


def _fixed_h(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    return inchi.MolToInchi(Chem.MolFromSmiles(smi), options="/FixedH /SNon")


def _key(smi):
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


def _exact(name, smiles):
    from tests.support.rt_assert import _independent_parse
    parsed = _independent_parse(name, smiles)
    return bool(parsed) and _key(parsed) == _key(smiles) and _fixed_h(parsed) == _fixed_h(smiles)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("one,other,pin", PINS)
def test_both_drawings_get_the_pin_with_the_lowest_prefix_locants(one, other, pin, tier):
    jar_or_skip()
    engine = _engine(tier)
    for smiles in (one, other):
        r = engine.name_tiered(smiles)
        assert (r["name"], r["tier"]) == (pin, "pin_verified"), (smiles, tier, r)
        assert _exact(pin, smiles), (smiles, pin)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("smiles,pin", NH_PLUS_DRAWING_ONLY)
def test_the_nn_azolium_drawn_with_the_charge_on_the_nh_nitrogen_gets_the_pin(smiles, pin, tier):
    jar_or_skip()
    r = _engine(tier).name_tiered(smiles)
    assert (r["name"], r["tier"]) == (pin, "pin_verified"), (smiles, tier, r)
    assert _exact(pin, smiles), (smiles, pin)


@pytest.mark.parametrize("one,other,pin", PINS)
def test_the_drawing_with_the_charge_on_the_nh_nitrogen_has_a_name(one, other, pin):
    """The producer, not the form choice: the NH+ drawing alone is named (the drawing is not one the
    rule of ``assembly.form_choice`` has to rescue with the other one)."""
    jar_or_skip()
    from rdkit import Chem

    from orthonym.rules.azolium_cations import azolium_hydron_site_name
    from orthonym.rules.ions import _ring_system_of
    mol = Chem.MolFromSmiles(one)
    cation = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetFormalCharge() == 1)
    assert azolium_hydron_site_name(mol, _ring_system_of(mol, cation), cation) == pin


@pytest.mark.parametrize("smiles", [
    "Nc1c[n+](C)cn1C",            # both nitrogens substituted, no NH+: azolium_demote_name's class
    "Nc1c[nH+]cs1",               # no substituted nitrogen: no indicated hydrogen to place
    "c1ccc2[nH+]c[nH]c2c1",       # two NH nitrogens: the benzimidazolium of the existing producers
    "Cc1c[nH+]c([N+](=O)[O-])n1C",  # another charged atom
])
def test_the_producer_declines_what_is_not_its_class(smiles):
    from rdkit import Chem

    from orthonym.rules.azolium_cations import azolium_hydron_site_name
    from orthonym.rules.ions import _ring_system_of
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        if a.GetSymbol() == "N" and a.GetFormalCharge() == 1 and a.IsInRing():
            ring = _ring_system_of(mol, a.GetIdx())
            assert azolium_hydron_site_name(mol, ring, a.GetIdx()) == "", (smiles, a.GetIdx())


def test_a_plain_azolium_keeps_the_name_the_other_producers_give():
    jar_or_skip()
    from rdkit import Chem

    from orthonym.rules.azolium_cations import azolium_hydron_site_name
    from orthonym.rules.ions import _ring_system_of
    mol = Chem.MolFromSmiles("Cc1c[nH+]cn1C")
    cation = next(a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() == 1 and a.GetSymbol() == "N")
    assert azolium_hydron_site_name(mol, _ring_system_of(mol, cation), cation) == "1,5-dimethyl-1H-imidazol-3-ium"
    for tier in TIERS:
        for smiles in ("Cc1c[nH+]cn1C", "Cc1c[nH]c[n+]1C"):
            r = _engine(tier).name_tiered(smiles)
            assert (r["name"], r["tier"]) == ("1,5-dimethyl-1H-imidazol-3-ium", "pin_verified"), (smiles, tier, r)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("smiles,pin", UNNAMED_RIVAL_KEEPS_THE_LABEL)
def test_a_name_keeps_pin_verified_when_the_other_charge_form_has_no_name(smiles, pin, tier):
    """Decision LO9: the exocyclic iminium drawing ('[NH2+]=C1NC(C(=O)[O-])CS1') is a cation that can only
    be a prefix, CATIONIC PREFIX NAMES (the Blue Book), and in a zwitterion
    INTRODUCTION (:42405),:42411, "anionic centers... become the parent structure, into which the
    cationic part is substituted"; the ring 'ium' holds its cation in the parent, CHOICE OF A
    PARENT STRUCTURE (:42330), (a) "parent with the maximum number of cationic, including cationic
    suffixes" (:42334). The Blue Book ranks the ring 'ium' name first whether or not the prefix name
    could be built, so the label stays."""
    jar_or_skip()
    r = _engine(tier).name_tiered(smiles)
    assert (r["name"], r["tier"]) == (pin, "pin_verified"), (smiles, tier, r)
    assert _exact(pin, smiles), (smiles, pin)
