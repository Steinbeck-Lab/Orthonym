"""v30: the enrichment path must use the vocabulary of the tier it is naming on.

``composer._integrate_universal_prefixes`` names every substituent for every
handler that enriches a core name, and it called::

    name_substituent(mol, sub_info.frag_atoms, attach_idx)

with no ``allow_mancude``. So enrichment used the PIN-default vocabulary even
when the molecule was being named on the **best-effort** tier, and refused
substituents the best-effort vocabulary can name -- measured, on the same mol /
fragment / attachment::

    NS(=O)=O                False -> 'substituent'   (a REFUSAL)
                            True  -> '1-amino-1-oxo-2-oxa-1lambda6-thiaeth-1-en-1-yl'
    c1ccc2c(c1)OCO2         False -> 'substituent'
                            True  -> '7,9-dioxabicyclo[4.3.0]nona-1,3,5-trien-4-yl'

⚠ **The hard constraint is why the flag arrives from the tier and not from the
call.** ``_integrate_universal_prefixes`` is shared by PIN-path handlers
(``acid_halides``, ``anhydrides``, ``esters``, ``lactones``, ``polyfunctional``
all call it), so threading ``allow_mancude=True`` unconditionally would change
PIN output and break hard bound H3 (PIN default byte-identical). None of those
handlers knows the tier, so the value is published once by the namer and read
here -- the same mechanism ``general_fallback_ctx`` already uses to reach
fragment recursion.

The discriminator is ``general_fallback_unverified`` (best-effort ONLY), NOT
``allow_aromatic_general`` -- that one is also True for ``complete``.
"""

import pytest
from rdkit import Chem

import orthonym.assembly.composer as comp
import orthonym.assembly.substituent_enumerator as se
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags


# --------------------------------------------------------------------------
# the contract: what value reaches name_substituent, per tier
# --------------------------------------------------------------------------

def _allow_mancude_values_for(smiles: str, tier: str, monkeypatch) -> list:
    """Every ``allow_mancude`` value ``_integrate_universal_prefixes`` passes to
    ``name_substituent`` while naming ``smiles`` on ``tier``.

    Records ONLY calls made from inside the enrichment helper: a
    ``name_substituent`` call on some other path is a different site and must
    not be credited to (or blamed on) this one.
    """
    seen = []
    depth = {"n": 0}
    orig_iup = comp._integrate_universal_prefixes
    orig_ns = se.name_substituent

    def spy_iup(*a, **k):
        depth["n"] += 1
        try:
            return orig_iup(*a, **k)
        finally:
            depth["n"] -= 1

    def spy_ns(mol, frag, attach, *a, **k):
        if depth["n"] > 0:
            am = k.get("allow_mancude")
            if am is None and len(a) >= 2:
                am = a[1]
            seen.append(bool(am))
        return orig_ns(mol, frag, attach, *a, **k)

    monkeypatch.setattr(comp, "_integrate_universal_prefixes", spy_iup)
    monkeypatch.setattr(se, "name_substituent", spy_ns)
    Orthonym(style="pin", **_emit_tier_flags(tier)).name(smiles)
    return seen


#: An acid halide with a branch. ``acid_halides`` calls the enrichment helper
#: explicitly, so this reaches the site -- verified against 6 such positives
#: before this test was written, because a spy that records zero for every input
#: proves nothing.
ENRICHING_INPUT = "CC(C)C(=O)Cl"


def test_the_spy_reaches_the_enrichment_site_at_all(monkeypatch):
    """Guard against a vacuous pass in every test below: if the helper stops
    calling ``name_substituent`` for this input, the per-tier assertions would
    all pass on an empty list."""
    seen = _allow_mancude_values_for(ENRICHING_INPUT, "best-effort", monkeypatch)
    assert seen, (
        "no name_substituent call was recorded from inside "
        "_integrate_universal_prefixes -- every other test here is vacuous")


def test_best_effort_enrichment_uses_the_best_effort_vocabulary(monkeypatch):
    seen = _allow_mancude_values_for(ENRICHING_INPUT, "best-effort", monkeypatch)
    assert all(seen), f"best-effort passed allow_mancude={seen}"


@pytest.mark.parametrize("tier", ["pin", "valid", "complete"])
def test_non_best_effort_enrichment_keeps_the_pin_vocabulary(tier, monkeypatch):
    """H3. ``complete`` is included deliberately: ``allow_mancude`` is not the
    best-effort discriminator, so a fix gated on ``allow_aromatic_general``
    would change ``complete`` output and this test would catch it."""
    seen = _allow_mancude_values_for(ENRICHING_INPUT, tier, monkeypatch)
    assert not any(seen), f"tier={tier} passed allow_mancude={seen}"


def test_the_flag_does_not_leak_between_namings(monkeypatch):
    """The value is published per naming call and must be torn down after it --
    a leak would silently put best-effort vocabulary on a later PIN naming in
    the same process, which is exactly the H3 break this is guarding."""
    assert all(_allow_mancude_values_for(
        ENRICHING_INPUT, "best-effort", monkeypatch))
    assert not any(_allow_mancude_values_for(
        ENRICHING_INPUT, "pin", monkeypatch))


# --------------------------------------------------------------------------
# the vocabulary difference this exists to exploit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frag_smiles,attach", [
    ("NS(=O)=O", 0),
    ("c1ccc2c(c1)OCO2", 0),
])
def test_the_two_flags_really_do_differ_for_these_fragments(frag_smiles, attach):
    """The premise, asserted rather than assumed. If a vocabulary fix later makes
    ``allow_mancude=False`` name these too, this test fails LOUDLY and tells the
    next reader the mechanism above no longer has anything to exploit -- rather
    than the row-level tests quietly passing for a different reason."""
    mol = Chem.MolFromSmiles(frag_smiles)
    assert mol is not None
    frag = list(range(mol.GetNumAtoms()))
    at_false = se.name_substituent(mol, frag, attach, allow_mancude=False)
    at_true = se.name_substituent(mol, frag, attach, allow_mancude=True)
    assert at_false in (None, "substituent"), (
        f"premise broken: allow_mancude=False now names {frag_smiles} as "
        f"{at_false!r}, so this fragment no longer demonstrates the gap")
    assert at_true not in (None, "substituent"), (
        f"premise broken: allow_mancude=True no longer names {frag_smiles}")


# --------------------------------------------------------------------------
# PIN byte-identity, at the level that actually matters
# --------------------------------------------------------------------------
# Captured from HEAD before the change. These all route through handlers that
# call the enrichment helper (acid halide, lactone, ester, anhydride), so they
# are the rows a mis-scoped flag would corrupt first.
PIN_UNCHANGED = [
    ("CC(C)C(=O)Cl", "2-methylpropanoyl chloride"),
    ("CC(Cl)C(=O)Cl", "2-chloropropanoyl chloride"),
    ("CCC(=O)Cl", "propanoyl chloride"),
    ("CC1CCC(=O)O1", "5-methyloxolan-2-one"),
    ("O=C1OCCC1", "oxolan-2-one"),
    ("CC(C)C(=O)OC", "methyl 2-methylpropanoate"),
    ("CC(=O)OC(C)=O", "acetic anhydride"),
    ("OCC(=O)C(=O)O", "3-hydroxy-2-oxopropanoic acid"),
]


@pytest.mark.parametrize("tier", ["pin", "complete"])
@pytest.mark.parametrize("smiles,expected", PIN_UNCHANGED)
def test_enriched_handler_names_are_unchanged_off_best_effort(
        tier, smiles, expected):
    assert Orthonym(style="pin", **_emit_tier_flags(tier)).name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", PIN_UNCHANGED)
def test_enriched_handler_names_are_unchanged_on_best_effort_too(
        smiles, expected):
    """These particular molecules have no substituent the two vocabularies
    disagree about, so widening the vocabulary must not disturb them either.
    Pins that the change is scoped to the refusal case, not a rewrite."""
    assert Orthonym(style="pin",
                     **_emit_tier_flags("best-effort")).name(smiles) == expected
