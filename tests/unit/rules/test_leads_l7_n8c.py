"""Leads program L7 / N8c: charge conservation for assembled organometallic names.

A dot-separated organometallic input whose ions do not sum to zero was named as the
NEUTRAL bonded complex and shipped at the default tier (label ``pin_unverified``,
through ``namer._ORGANOMETALLIC_ADDITIVE_PIN_RE``, which bypasses OPSIN), e.g.
``[Fe].c1cc[cH-]c1.c1cc[cH-]c1`` (net -2) -> 'bis(eta5-cyclopentadienyl)iron'.

Since 2026-11-01 the gate ships no eta name at any tier (no read-back;, the Blue Book); these tests read
the raw producer (the validity gate is off under pytest without the opsin_gate mark), whose charge check still matters.

Governing rule: is out of scope for the engine; the rule that applies is core
principle 2 (never emit a wrong structure). The Blue Book states a charged complex
with a charge number: 'Compounds with at least one metal-carbon single
bond' (the Blue Book), example:39793 'pentaammine(ethanido)osmium(1+)
chloride'; 'Organometallic groups with multicenter bonding to carbon
atoms' (:39799), example:40104
'tricarbonyl(eta7-cycloheptatrienylium)molybdenum(1+)'. A name with no charge
number describes a neutral compound.

The independent check (does not use the code under test) is RDKit's formal charge
of the input: every declined row has a non-zero net charge, every kept control is
net zero. Balanced forms are NOT wrong: ``c1ccccc1.[Cl-].[Cl-].[Fe+2]`` has the same
standard InChIKey as ``c1ccccc1.Cl[Fe]Cl`` and dot-separated pi-ligands are the
SMILES convention for sandwich complexes.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _emit_tier_flags
from orthonym.namer import Orthonym, _validity_gate_name_to_smiles
from orthonym.errors import is_failure_name
from orthonym.rules.organometallics import (
    organometallic_charge_conserved,
    stated_charge,
)

# (smiles, net charge) -- every one used to ship a neutral-complex name.
CHARGED = [
    ("c1ccccc1.[Cl-].[Cl-].[Fe-3]", -5),
    ("c1ccccc1.[Cl-].[Cl-].[Fe]", -2),
    ("[Cr+].c1ccccc1.c1ccccc1", 1),
    ("[Fe].c1cc[cH-]c1.c1cc[cH-]c1", -2),
    ("[Ti].[Cl-].[Cl-].c1cc[cH-]c1.c1cc[cH-]c1", -4),
    ("[Cr].c1ccccc1.[C-]#[O+].[C-]#[O+].[Cl-]", -1),
    ("C1=CC=CC=CC=C1.[Ni+2]", 2),
    ("[Mn].c1cc[cH-]c1.[C-]#[O+].[C-]#[O+].[C-]#[O+]", -1),
]

# (smiles, expected name) -- net zero, must keep their names.
BALANCED = [
    ("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1", "ferrocene"),
    ("c1ccccc1.[Cl-].[Cl-].[Fe+2]", "(η⁶-benzene)bis(η¹-chlorido)iron"),
    ("[Cr].c1ccccc1.c1ccccc1", "bis(η⁶-benzene)chromium"),
    (
        "[Mn+].c1cc[cH-]c1.[C-]#[O+].[C-]#[O+].[C-]#[O+]",
        "tricarbonyl(η⁵-cyclopentadienyl)manganese",
    ),
    ("[Ti+4].[Cl-].[Cl-].c1cc[cH-]c1.c1cc[cH-]c1",
     "bis(η¹-chlorido)bis(η⁵-cyclopentadienyl)titanium"),
]


@pytest.mark.unit
class TestN8cChargeConservation:
    @pytest.mark.parametrize("smiles,charge", CHARGED)
    def test_input_is_charged(self, smiles, charge):
        # independent of the code under test
        assert Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)) == charge

    @pytest.mark.parametrize("smiles,charge", CHARGED)
    def test_charge_imbalanced_input_is_not_named_as_neutral_complex(self, smiles, charge):
        r = Orthonym(style="pin").name_tiered(smiles)
        name = r.get("name") or ""
        # no assembled eta / -ido complex name may ship for a charged input
        assert not any(t in name for t in ("η", "chlorido", "carbonyl")), (smiles, name)
        assert r["tier"] == "abstain", (smiles, name, r["tier"])

    @pytest.mark.parametrize("smiles,expected", BALANCED)
    def test_balanced_input_keeps_its_name(self, smiles, expected):
        assert Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)) == 0
        assert Orthonym(style="pin").name_tiered(smiles)["name"] == expected


@pytest.mark.unit
class TestChargePredicate:
    @pytest.mark.parametrize(
        "name,stated",
        [
            ("ferrocene", ("net", 0)),
            ("tricarbonyliron", ("net", 0)),
            ("tetracarbonylferrate(1-)", ("net", -1)),
            ("tetracarbonyl(η²-ethene)iron(2+)", ("net", 2)),
            # a Stock number is the oxidation number of the metal, not a net charge
            ("bis(η⁵-cyclopentadienyl)iron(III)", ("oxidation", 3)),
            ("hexacarbonylvanadium(-I)", ("oxidation", -1)),
            ("hexacarbonylchromium(0)", ("oxidation", 0)),
            ("pentaammine(ethyl)osmium(1+) chloride", ("net", 0)),  # not the closing word
        ],
    )
    def test_stated_charge(self, name, stated):
        assert stated_charge(name) == stated

    def test_net_charge_must_equal_the_input_charge(self):
        neutral = Chem.MolFromSmiles("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1")
        anion = Chem.MolFromSmiles("[Fe].c1cc[cH-]c1.c1cc[cH-]c1")
        assert organometallic_charge_conserved("bis(η⁵-cyclopentadienyl)iron", neutral)
        assert not organometallic_charge_conserved("bis(η⁵-cyclopentadienyl)iron", anion)
        assert organometallic_charge_conserved("bis(η⁵-cyclopentadienyl)iron(2-)", anion)

    def test_stock_number_must_equal_the_metal_charge(self):
        # ferrocenium, net +1: Fe(III) with two Cp(-): the Stock name is right
        cation = Chem.MolFromSmiles("[Fe+3].c1cc[cH-]c1.c1cc[cH-]c1")
        anion = Chem.MolFromSmiles("[Fe].c1cc[cH-]c1.c1cc[cH-]c1")
        assert organometallic_charge_conserved("bis(η⁵-cyclopentadienyl)iron(III)", cation)
        # Fe(0) with two Cp(-): the default Stock number of the systematic style (II) is
        # a metal charge the input does not have
        assert not organometallic_charge_conserved("bis(η⁵-cyclopentadienyl)iron(II)", anion)
        vanadate = Chem.MolFromSmiles("[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[V-]")
        assert organometallic_charge_conserved("hexacarbonylvanadium(-I)", vanadate)


@pytest.mark.unit
class TestSystematicStyle:
    def test_stock_names_of_charged_complexes_keep_their_names(self):
        assert (
            Orthonym(style="systematic").name("[Fe+3].c1cc[cH-]c1.c1cc[cH-]c1")
            == "bis(η⁵-cyclopentadienyl)iron(III)"
        )
        assert (
            Orthonym(style="systematic").name(
                "[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[V-]")
            == "hexacarbonylvanadium(-I)"
        )

    def test_charge_imbalanced_input_is_declined(self):
        # Fe(0) + 2 Cp(-) is net -2; 'iron(II)' would state another metal charge
        name = Orthonym(style="systematic").name("[Fe].c1cc[cH-]c1.c1cc[cH-]c1")
        assert is_failure_name(name), name

    def test_cymantrene_neutral_and_anion(self):
        # canary ORG-T4-03: the row was drawn with a neutral Mn (Mn(0) + Cp(-), the ANION,
        # net -1) although both of its names describe neutral Mn(I) cymantrene
        neutral = "[C-]#[O+].[C-]#[O+].[C-]#[O+].[Mn+].c1cc[cH-]c1"
        anion = "[C-]#[O+].[C-]#[O+].[C-]#[O+].[Mn].c1cc[cH-]c1"
        assert Chem.GetFormalCharge(Chem.MolFromSmiles(neutral)) == 0
        assert Chem.GetFormalCharge(Chem.MolFromSmiles(anion)) == -1
        assert (Orthonym(style="pin").name(neutral)
                == "tricarbonyl(η⁵-cyclopentadienyl)manganese")
        assert (Orthonym(style="systematic").name(neutral)
                == "tricarbonyl(η⁵-cyclopentadienyl)manganese(I)")
        assert is_failure_name(Orthonym(style="pin").name(anion))
        assert is_failure_name(Orthonym(style="systematic").name(anion))


# ---------------------------------------------------------------------------
# The net charge a name states is the formal charge of the structure OPSIN reads
# from it, so a charge stated by a suffix ('-ate') or a prefix ('sulfonato') counts
# as much as a closing Ewens-Bassett number. The first version of the check read only
# the closing number and declined correct names whose charge sits in a ligand:
# 'O=S(=O)([O-])c1cc[c]([Hg][Cl])cc1' lost '(4-sulfonatophenyl)mercury chloride'
# (ChEBI, best-effort tier; it round-trips exactly). A name OPSIN cannot read (the eta
# and additive names the pin tier ships past OPSIN) still falls back to the closing
# number, which the tests above hold.
# Governing rule: 'Anions derived from acids' (the Blue Book,
# 'benzoate', 'sulfonate') and 'Cationic centers on characteristic groups'
# (:41413) state a charge in the name; core principle 2 (never emit a wrong structure)
# is the reason the check exists.
# ---------------------------------------------------------------------------

HG_SULFONATE = "O=S(=O)([O-])c1cc[c]([Hg][Cl])cc1"
HG_SULFONATE_NAME = "(4-sulfonatophenyl)mercury chloride"


def _best_effort(smiles):
    return Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)


def _round_trips_exactly(smiles, name):
    """OPSIN's reading of ``name`` is the input: the same canonical SMILES and the same
    full InChIKey (independent of the code under test)."""
    parsed = _validity_gate_name_to_smiles(name)
    assert parsed, f"OPSIN cannot read {name!r}"
    want, got = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(parsed)
    assert Chem.MolToSmiles(got) == Chem.MolToSmiles(want), (name, parsed)
    assert Chem.MolToInchiKey(got) == Chem.MolToInchiKey(want), (name, parsed)


@pytest.mark.unit
class TestChargeStatedByTheNameIsReadByOpsin:
    def test_sulfonato_mercury_row_keeps_its_best_effort_name(self):
        assert Chem.GetFormalCharge(Chem.MolFromSmiles(HG_SULFONATE)) == -1
        r = _best_effort(HG_SULFONATE)
        assert r["name"] == HG_SULFONATE_NAME, r
        assert r["tier"] != "abstain"
        _round_trips_exactly(HG_SULFONATE, r["name"])

    @pytest.mark.parametrize(
        "name,smiles,conserved",
        [
            # the charge is stated by a prefix
            ("(4-sulfonatophenyl)mercury chloride", HG_SULFONATE, True),
            ("(2-sulfonatoethyl)mercury chloride", "[O-]S(=O)(=O)CC[Hg]Cl", True),
            ("(carboxylatomethyl)(methyl)mercury", "C[Hg]CC(=O)[O-]", True),
            # the charge is stated by a suffix
            ("4-(chloromercurio)benzoate", "[O-]C(=O)c1ccc([Hg]Cl)cc1", True),
            # a name that states a neutral acid does not state the anion's charge
            ("(4-sulfophenyl)mercury chloride", HG_SULFONATE, False),
            ("(carboxymethyl)(methyl)mercury", "C[Hg]CC(=O)[O-]", False),
            ("4-(chloromercurio)benzoic acid", "[O-]C(=O)c1ccc([Hg]Cl)cc1", False),
            # a name that states an anion does not describe the neutral acid
            ("(4-sulfonatophenyl)mercury chloride",
             "O=S(=O)(O)c1ccc([Hg]Cl)cc1", False),
            ("4-(chloromercurio)benzoate", "OC(=O)c1ccc([Hg]Cl)cc1", False),
        ],
    )
    def test_charge_of_the_name_equals_charge_of_the_input(self, name, smiles, conserved):
        assert organometallic_charge_conserved(name, Chem.MolFromSmiles(smiles)) is conserved

    def test_a_name_opsin_cannot_read_states_its_charge_by_its_closing_number(self, monkeypatch):
        # the eta / additive names OPSIN cannot read: the Ewens-Bassett number decides
        from orthonym.rules import organometallics as om
        monkeypatch.setattr(om, "_opsin_read_net_charge", lambda name: None)
        anion = Chem.MolFromSmiles(HG_SULFONATE)
        neutral = Chem.MolFromSmiles("O=S(=O)(O)c1ccc([Hg]Cl)cc1")
        assert organometallic_charge_conserved("(4-sulfophenyl)mercury chloride", neutral)
        assert not organometallic_charge_conserved(HG_SULFONATE_NAME, anion)
        chromium = Chem.MolFromSmiles("[Cr]" + ".[C-]#[O+]" * 6)
        assert organometallic_charge_conserved("hexacarbonylchromium(0)", chromium)


@pytest.mark.unit
class TestArsenicRowIsNotACauseOfThisCheck:
    """The second ChEBI row of the N8c report, 'O=[N+]([O-])c1cc([As](O)O)ccc1[O-]'
    ('4-(dihydroxyarsanyl)-2-nitrophenolate', pin_verified before), is NOT declined by the
    charge check: the organometallic handler is never reached for it (the arsenic is an
    As-C acid, named by the pnictogen-oxoacid handler) and bisecting that row alone gives
    2053a316d (item 42, 'an oxoacid with a substitutable hydrogen is a functional parent'),
    whose nitro exemption in rules.phosphorus._organyl_or_junior_prefix lets the charged
    router name the neutralised skeleton '(4-hydroxy-3-nitrophenyl)arsonous acid', a name
    from which it cannot rebuild the phenolate. The class was wider than the row: every
    mono-anion of a hydroxy or thiol group whose neutral skeleton has a senior acid, amide,
    nitrile, aldehyde or ketone abstained at the default tier ('4-carboxyphenolate' too). The
    router now names the anion on its own group: anions are senior to acids;
    ``charged_router.olate_with_senior_neutral_group``): a benzene ring, a chain, an aliphatic
    ring and a heteroaromatic ring, with the members it does not build pinned as strict
    expected failures in ``tests/unit/rules/test_olate_senior_neutral_group.py``."""

    def test_nitrophenolate_with_arsonous_acid_prefix_is_named(self):
        smiles = "O=[N+]([O-])c1cc([As](O)O)ccc1[O-]"
        name = "4-(dihydroxyarsanyl)-2-nitrophenolate"
        r = _best_effort(smiles)
        assert r["name"] == name, r
        _round_trips_exactly(smiles, name)
