""" — the anionic centre outranks the cationic centre for LOW LOCANTS.

 residue Task G. The residue item recorded " anion locants" as an
unbuilt gap in the general engine. **The premise is refuted**: the rule is
implemented, it is byte-correct on every Blue-Book-documented example of the
class, and the general engine is not even on the path — the producer is
``rules/ions.py::emit_cumulative_ium_ide`` (its ``_score`` helper), reached via
``routing/dispatch_table.py``'s ``CUMULATIVE_ZWITTERION`` entry. A runtime trace
on ``assembly/general_engine.py``'s three charge sites
(``_charge_suffix_text`` / ``_append_charge_suffix`` / ``_zwitterion_suffix_plan``)
recorded ZERO calls for every molecule below, with the trace validated against a
known positive (``C[n+]1ccccc1C(=O)[O-]`` routes through
``rules/charged_router.py::_route_zwitterion``) and a negative control
(``CCO`` -> 0 calls).

This file exists so the refutation cannot silently rot: these names are the
evidence, and they are currently pinned nowhere else except one gold row.

Blue Book, ``the Blue Book Blue Book`` (every pointer re-opened with
``sed -n '<N>p'`` at write time):

* ** "INTRODUCTION"**, sentence ``:42411``: *"According to the seniority
  of classes, an anionic center has priority over a cationic center in
  zwitterions. Thus, in zwitterionic compounds anionic centers are preferred
  for lower locants and become the parent structure, into which the cationic
  part is substituted. CAS gives cationic centers priority over anionic
  centers."*
* ** "Ionic centers in the same parent structure"** (heading
  ``:42415``), sentence ``:42417``: *"…anionic suffixes are cited after
  cationic suffixes in the name, and are given seniority for low locants. …
  Where there is a choice, lowest locants are given to the ionic centers in the
  following order, listed in decreasing order of seniority: 'uide' ('uida'),
  'ide' ('ida'), 'ylium' ('ylia'), and 'ium' ('onia')."*

Note the last clause is what makes this testable at all: the rule is only
observable on a molecule where the anion-low numbering **contradicts** some
other low-locant criterion. Two such molecules are included below and marked
DISCRIMINATOR; on both, giving the cation the low locant would produce lower
substituent locants, and the Blue Book still requires the anion to win.
"""

import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# (SMILES, expected PIN, Blue Book provenance)
#
# The first four are the Blue Book's own worked (PIN) examples for the
# cumulative same-parent zwitterion. The last two are DISCRIMINATORS
# constructed for this test: on each, the anion-low numbering is the
# one that does NOT minimise the substituent locant set, so a producer that
# ranked substituents first would emit the other name.
P74_CUMULATIVE_ZWITTERIONS = [
    (
        "C[N-][N+](C)(C)C",
        "1,2,2,2-tetramethylhydrazin-2-ium-1-ide",
        "P-74.1.1 :42423 (PIN). DISCRIMINATOR: anion-low gives methyl locants "
        "{1,2,2,2}; cation-low would give the LOWER set {1,1,1,2}. The Blue "
        "Book prints 1,2,2,2 -- so P-74.1.1 anion seniority beats P-14.4 "
        "substituent locants.",
    ),
    (
        "C[N+]([N-]C)=NC",
        "1,2,3-trimethyltriaz-2-en-2-ium-1-ide",
        "P-74.2.2.1.1 azo imides :43121 (PIN); numbering by P-74.1.1 :42417.",
    ),
    (
        "C[N-][N+](=C)C",
        "1,2-dimethyl-2-methylidenehydrazin-2-ium-1-ide",
        "P-74.2.2.1.2 azomethine imides :43143 (PIN).",
    ),
    (
        "CC(C)=[O+][O-]",
        "2-(propan-2-ylidene)dioxidan-2-ium-1-ide",
        "P-74.2.2.1.6 carbonyl oxides :43187 (PIN). Confirms the rule is not "
        "nitrogen-specific -- the same anion-low ordering applies on a "
        "homogeneous OXYGEN chain parent hydride (dioxidane).",
    ),
    (
        "C[N+](C)(C)N[N-]C",
        "1,3,3,3-tetramethyltriazan-3-ium-1-ide",
        "DISCRIMINATOR (triazane, 3 skeletal N). Anion-low puts the '-ide' at "
        "1 and the '-ium' at 3, giving methyl locants {1,3,3,3}; cation-low "
        "would give the LOWER set {1,1,1,3}. P-74.0 :42411 + P-74.1.1 :42417 "
        "require the anion at 1.",
    ),
    (
        "CC[N-][N+](C)(C)C",
        "1-ethyl-2,2,2-trimethylhydrazin-2-ium-1-ide",
        "DISCRIMINATOR (unsymmetrical substitution). Anion-low fixes the "
        "ethyl at 1 and the three methyls at 2; it also fixes which end the "
        "alphabetised prefixes attach to, so a reversed numbering would be "
        "visible in BOTH the locants and the prefix order.",
    ),
]


@pytest.mark.parametrize(
    "smiles,expected,provenance",
    P74_CUMULATIVE_ZWITTERIONS,
    ids=[s for s, _, _ in P74_CUMULATIVE_ZWITTERIONS],
)
def test_anion_takes_the_low_locant(namer, smiles, expected, provenance):
    """:42417 -- the '-ide' locant is minimised before the '-ium'."""
    assert namer.name(smiles) == expected, provenance


def test_the_ide_locant_is_lower_than_the_ium_locant(namer):
    """:42411 -- 'anionic centers are preferred for lower locants'.

    A structural restatement of the rule that does not depend on any single
    spelling: across the whole class, the ``-ide`` locant must never exceed the
    ``-ium`` locant. This catches a numbering regression even if the stem,
    substituent prefixes or elision were to change.
    """
    import re

    checked = 0
    for smiles, _expected, _prov in P74_CUMULATIVE_ZWITTERIONS:
        name = namer.name(smiles)
        m = re.search(r"-(\d+)-ium-(\d+)-ide$", name)
        assert m is not None, (
            "expected a cumulative '-<n>-ium-<n>-ide' ending, got %r for %s"
            % (name, smiles)
        )
        ium_locant, ide_locant = int(m.group(1)), int(m.group(2))
        assert ide_locant < ium_locant, (
            "P-74.0 :42411 violated for %s: '%s' puts the anion at %d and the "
            "cation at %d" % (smiles, name, ide_locant, ium_locant)
        )
        checked += 1
    # Guard against a vacuous pass if the parametrised list is ever emptied.
    assert checked == len(P74_CUMULATIVE_ZWITTERIONS) >= 6


def test_internal_charge_groups_are_not_read_as_zwitterions(namer):
    """ / -- nitro and azido carry formal +/- that are BONDING
    features, not ionic centres.

    The negative half of the class. Without this boundary the azide's central
    N+ / terminal N- are mis-read as a triazene zwitterion and azidobenzene is
    emitted as '3-phenyltriaz-1,2-dien-2-ium-1-ide'. Kept here because it is
    the guard that makes the positive cases above safe to widen.
    """
    assert namer.name("O=[N+]([O-])c1ccccc1") == "nitrobenzene"
    assert namer.name("[N-]=[N+]=Nc1ccccc1") == "azidobenzene"
