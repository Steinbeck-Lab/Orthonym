"""Locant assignment / '1'-citation conformance +.

These pin the SPELLING of locants on names the engine already emits for the
right molecule. Every assertion additionally proves, via OPSIN round-trip, that
the re-cited name denotes the SAME structure as the input (a pure spelling fix,
never a structure change).

Governing rules (the Blue Book Blue Book):
  * "Citation of locants" (the Blue Book) -- DENY-BY-DEFAULT: once any locant
    in a scope is essential, EVERY locant in that scope is cited.
  * "Omission of locants" (the Blue Book) -- six enumerated licences that
    permit dropping a locant, evaluated per enclosing-mark scope.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound

pytestmark = pytest.mark.unit


def _inchikey(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _opsin_same_structure(name: str, smiles: str) -> bool:
    """True iff OPSIN parses `name` to the same InChIKey as `smiles`."""
    from orthonym.validation.opsin_roundtrip import opsin_parse

    parsed = opsin_parse(name)
    if not parsed:
        return False
    return _inchikey(parsed) == _inchikey(smiles)


# ---------------------------------------------------------------------------
# Sub-fix (ii), OMIT direction: a skeletal-replacement heterocycle that bears a
# numbered suffix MUST cite the heteroatom locant '1' deny-default).
# The '-2-one' suffix cites locant 2, so 'oxa'/'aza' at position 1 is cited too.
# Before the fix the engine emitted 'oxacyclododecan-2-one' (locant '1' dropped).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected",
    [
        # [BB 65.6.3.5.1] macrocyclic lactone, replacement nomenclature.
        ("O=C1CCCCCCCCCCO1", "1-oxacyclododecan-2-one"),
        # [BB 66.1.5.1] macrocyclic lactam, replacement nomenclature.
        ("O=C1CCCCCCCCCCCN1", "1-azacyclotridecan-2-one"),
        # 11-membered lactone -- same class, ring one smaller.
        ("O=C1CCCCCCCCCO1", "1-oxacycloundecan-2-one"),
    ],
)
def test_replacement_ring_cites_heteroatom_locant_one(smiles, expected):
    got = name_compound(smiles)
    assert got == expected, f"{smiles}: got {got!r}, want {expected!r}"
    assert _opsin_same_structure(got, smiles), (
        f"{got!r} must round-trip to the input structure"
    )


def test_bare_replacement_parent_still_omits_locant_one():
    """The BARE parent hydride (no numbered suffix) keeps the locant omitted --
    : nothing else is numbered, so 'thiacyclotridecane' NOT
    '1-thiacyclotridecane'. Guards against over-citing after the OMIT fix."""
    got = name_compound("C1CCCCCCCCCCCS1")  # 13-membered, one S
    assert got == "thiacyclotridecane", f"got {got!r}"


# ---------------------------------------------------------------------------
# Substituent-join hardening: when the parent hydride itself begins with a
# locant, a substituent prefix is separated from it by a hyphen
# letter-to-locant boundary): '3-methyl-1-oxacyclododecan-2-one', never
# '3-methyl1-oxacyclododecan-2-one'.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("O=C1C(C)CCCCCCCCCO1", "3-methyl-1-oxacyclododecan-2-one"),
        ("O=C1CC(C)CCCCCCCCCN1", "4-methyl-1-azacyclotridecan-2-one"),
        # Substituent on the numbered ring N of a macrolactam: the
        # ring N is locant 1, so N-methyl is cited '1-methyl' and merges with the
        # parent's '1-aza' -> '1-methyl-1-azacyclotridecan-2-one'.
        ("O=C1N(C)CCCCCCCCCCC1", "1-methyl-1-azacyclotridecan-2-one"),
    ],
)
def test_substituted_replacement_ring_hyphenates_prefix(smiles, expected):
    got = name_compound(smiles)
    assert got == expected, f"{smiles}: got {got!r}, want {expected!r}"
    assert _opsin_same_structure(got, smiles), (
        f"{got!r} must round-trip to the input structure"
    )
