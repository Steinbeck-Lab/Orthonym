"""
M6 Kind-C — 2-ring fused-heterocycle catalog expansion.

internal notes and M6-KINDB-CONFIRM.md measured that the
composer's fused-ring routing (composer.py's ortho-fused branch calling
``name_fused_heterocycle``, and ``rules.spiro._name_fused_component`` for the
spiro-mixed path) is ALREADY correct and gated only on
``match_fused_heterocycle_core`` hitting the ``FUSED_HETEROCYCLE_DATA``
catalog in ``data/fused_heterocycles.py``. 23/44 spiro-mixed witnesses and a
further slice of the plain ortho-fused witnesses failed with
``fused_component_uncatalogued`` -- the 2-ring fused core was simply absent
from the catalog, not a routing/detector bug.

This module pins the four 2-ring cores added to close that gap, each of
which was OPSIN-round-trip-validated (constitution-only InChI match to the
exact bare core, never a hand-guessed name) before being added:

  * 9H-fluorene (tricyclic PAH)
  * 3,4-dihydro-2H-1,4-benzoxazine (benzomorpholine)
  * pyrazolo[1,5-a]pyrazine (bridgehead-N 5-6)
  * 5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole (5-5 heteroaromatic)

Both the bare parents and small substituted derivatives are pinned, plus two
of the real corpus witnesses that measurably converted from
``unknown organic compound`` to a full OPSIN-round-tripping name once the
catalog held their core (see internal notes for
the full before/after and the two witnesses that still abstain for reasons
unrelated to this catalog addition).
"""

import pytest
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi

from orthonym import name_compound
from orthonym.data.fused_heterocycles import match_fused_heterocycle_core


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # 9H-fluorene: bare parent, and substituted (indicated H suppressed
        # per the existing engine convention once the ring is substituted
        # only in the way that keeps 9H unambiguous -- OPSIN round-trip is
        # the authority checked below, not this string).
        # (the Blue Book-11400): '(9H-isomer shown; the PIN is 9H-fluorene)'
        ("C1c2ccccc2-c2ccccc21", "9H-fluorene"),
        # 3,4-dihydro-2H-1,4-benzoxazine: bare parent.
        ("C1COc2ccccc2N1", "3,4-dihydro-2H-1,4-benzoxazine"),
        # pyrazolo[1,5-a]pyrazine: bare parent and a methyl derivative
        # (confirms the bridgehead-N / lettered-carbon locant map generalizes
        # to substituent placement, not just the unsubstituted lookup).
        ("c1cn2nccc2cn1", "pyrazolo[1,5-a]pyrazine"),
        ("Cc1cn2nccc2cn1", "6-methylpyrazolo[1,5-a]pyrazine"),
        # 5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole: bare parent
        # and a 6-phenyl derivative.
        ("c1nnc2n1NCS2", "5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole"),
        (
            "c1nnc2n1NC(c3ccccc3)S2",
            "6-phenyl-5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole",
        ),
    ],
)
def test_m6_kindc_catalog_entries_name(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles",
    [
        "C1c2ccccc2-c2ccccc21",
        "C1COc2ccccc2N1",
        "c1cn2nccc2cn1",
        "Cc1cn2nccc2cn1",
        "c1nnc2n1NCS2",
        "c1nnc2n1NC(c3ccccc3)S2",
    ],
)
def test_m6_kindc_catalog_entries_roundtrip(smiles):
    """Every pinned name OPSIN-round-trips to the exact input structure
    (0-wrong gate) -- constitution AND full InChI, since none of these carry
    stereocentres."""
    from orthonym.validation.opsin_roundtrip import opsin_parse

    name = name_compound(smiles)
    opsin_smiles = opsin_parse(name)
    assert opsin_smiles is not None, f"OPSIN rejected {name!r}"

    orig = Chem.MolFromSmiles(smiles)
    parsed = Chem.MolFromSmiles(opsin_smiles)
    assert orig is not None and parsed is not None
    assert MolToInchi(orig) == MolToInchi(parsed)


@pytest.mark.unit
def test_m6_kindc_catalog_covers_the_four_new_cores():
    """Direct catalog-lookup check (bypasses the full naming pipeline) for
    the four cores this module adds -- the exact ``match_fused_heterocycle_core``
    gate the M6 trace docs identify as the self-activating choke point."""
    cases = [
        ("C1c2ccccc2-c2ccccc21", "9H-fluorene"),
        ("C1COc2ccccc2N1", "3,4-dihydro-2H-1,4-benzoxazine"),
        ("c1cn2nccc2cn1", "pyrazolo[1,5-a]pyrazine"),
        ("c1nnc2n1NCS2", "5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole"),
    ]
    for smiles, expected_name in cases:
        mol = Chem.MolFromSmiles(smiles)
        result = match_fused_heterocycle_core(mol)
        assert result is not None, f"{smiles} still uncatalogued"
        assert result[0] == expected_name


@pytest.mark.unit
def test_m6_kindc_real_witnesses_reach_the_catalog():
    """The catalog match itself (deterministic, structural-only -- no
    substituent-name assembly) now succeeds directly on the WHOLE real
    corpus witness molecules, for the three non-spiro-mixed cores. This is
    the specific ``match_fused_heterocycle_core`` gate the M6-FUSED-a trace /
    M6-KINDB-CONFIRM docs identify as the self-activating choke point, and
    is the deterministic half of the claim (see
    test_m6_kindc_real_witnesses_convert for the full-pipeline half).

    The fourth real witness (the benzoxazine one) is spiro-mixed (Kind B):
    ``match_fused_heterocycle_core`` correctly declines on the WHOLE
    molecule there (the local ring system it touches extends past the bare
    benzoxazine core into the spiro side ring, so the
    ``_match_covers_ring_systems`` guard legitimately vetoes a partial
    match) -- ``rules.spiro._name_fused_component`` extracts the fused-ring
    fragment FIRST and matches the catalog against that isolated fragment
    instead, which is exercised end-to-end by
    test_m6_kindc_real_witnesses_convert."""
    cases = [
        # 9H-fluorene core (this witness's FULL pipeline name is pinned
        # separately below only via the catalog-match check, not the final
        # string -- see the report for a pytest-environment-only
        # nondeterminism found in its unrelated CF3-sulfonyl-imine
        # substituent naming, orthogonal to this catalog addition).
        ("O=S(=O)(ON=C(c1ccc2c(c1)Cc1ccccc1-2)C(F)(F)F)C(F)(F)F", "9H-fluorene"),
        # pyrazolo[1,5-a]pyrazine core -- reached, though the overall
        # molecule still abstains for an unrelated reason (a spirocyclobutane
        # -nitrile substituent-naming bug rejected by self_consistency; not
        # this catalog -- see the report).
        (
            "Cn1ccc(-c2cn3nccc3c(-c3cnn(C4CC5(CC(C#N)C5)C4)c3)n2)n1",
            "pyrazolo[1,5-a]pyrazine",
        ),
        # 5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole core --
        # reached, though the overall molecule still abstains for an
        # unrelated reason (a nitro-group tautomer/charge form the
        # self-consistency gate does not equate with the neutral form OPSIN
        # emits; not this catalog -- see the report).
        (
            "[O-][NH+](O)c1ccc([C@@H]2Nn3c(nnc3-c3cccnc3)S2)cc1",
            "5,6-dihydro-[1,2,4]triazolo[3,4-b][1,3,4]thiadiazole",
        ),
    ]
    for smiles, expected_name in cases:
        mol = Chem.MolFromSmiles(smiles)
        result = match_fused_heterocycle_core(mol)
        assert result is not None, f"{smiles} still uncatalogued"
        assert result[0] == expected_name


@pytest.mark.unit
@pytest.mark.roundtrip
def test_m6_kindc_real_witnesses_convert():
    """A real M6-trace corpus witness (the spiro-mixed / Kind-B benzoxazine
    case) that measurably converted from ``unknown organic compound`` to a
    full OPSIN-round-tripping name once its fused core was cataloged.

    Only this one witness is pinned end-to-end here: the other three real
    witnesses this catalog addition reaches (fluorene, pyrazolopyrazine,
    triazolothiadiazole -- see test_m6_kindc_real_witnesses_reach_the_catalog)
    either still abstain for a documented UNRELATED reason (a separate
    substituent-naming/self-consistency issue on a distant fragment, not
    this catalog), or -- the fluorene witness -- were found during
    verification to emit a nondeterministic (pytest-environment-only)
    wrong-molecule name from that same unrelated CF3-sulfonyl-imine
    substituent path. Both are reported as residual findings rather than
    pinned here, since asserting on them would test unrelated, currently
    unreliable machinery instead of this catalog change."""
    from orthonym.namer import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_parse

    smiles = "CN1CCC(n2cc(-c3ccc4c(c3)OCC3(C=C(CN5CCOCC5)N3)N4)cn2)CC1"
    engine = Orthonym(
        general_fallback=True,
        general_fallback_unverified=True,
        allow_aromatic_general=True,
    )
    name = engine.name(smiles)
    assert name != "unknown organic compound", (
        f"{smiles} still abstains after the M6 Kind-C catalog addition"
    )
    opsin_smiles = opsin_parse(name)
    assert opsin_smiles is not None, f"OPSIN rejected {name!r}"

    orig = Chem.MolFromSmiles(smiles)
    parsed = Chem.MolFromSmiles(opsin_smiles)
    assert orig is not None and parsed is not None
    assert MolToInchi(orig) == MolToInchi(parsed), (
        f"{smiles} -> {name!r} does not round-trip to the input structure"
    )
