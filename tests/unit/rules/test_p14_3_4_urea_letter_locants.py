"""Urea locants: MONO omits, DI keeps, chalcogen analogues keep — a tripwire.

 Phase C Task 7(a), CORRECTED. This file exists so a change to any
 licence cannot break either half of the urea boundary: a MONOsubstituted
urea must OMIT its italic-N locant, while a DIsubstituted urea and every CHALCOGEN
analogue must KEEP theirs.

⚠ THIS FILE'S ORIGINAL THESIS WAS BACKWARDS AND IS NOW CORRECTED
----------------------------------------------------------------
It formerly asserted ``CNC(=O)N -> N-methylurea`` and argued that ``methylurea``
(``the Blue Book Blue Book``) was outranked by. That reading
was wrong, and the oracle-integrity pass corrected the gold to ``methylurea``.
The resolution:

* ``:2943`` prints, verbatim, inside the example block of §**** ("The
  locant is omitted in monosubstituted symmetrical parent hydrides or parent
  compounds where there is only one kind of substitutable hydrogen")::

      CH3-NH-CO-NH2 methylurea (PIN)

  Urea's four N-H are a single ``CanonicalRankAtoms(breakTies=False)`` orbit, so a
  single N-substituent is unambiguous and the locant is omitted.
* §**** (``:33308``) — "urea... is the preferred IUPAC name, **with
  locants N and N'**" and ``:33318`` "Numerical locants for urea are no longer used
  in the IUPAC preferred name" — define urea's locant SCHEME (letter ``N``/``N'``,
  NOT numeric ``1``/``2``/``3``). They say which locant to cite *when one is cited*;
  they do NOT mandate citing one. Every urea example in that block that DOES cite a
  locant is DIsubstituted (``:33327`` ``N,N'-dimethylurea (PIN)``, ``:33336``
  ``N-[1-cyano-3-(methylsulfanyl)propyl]-N'-methylurea (PIN)``), where the locant is
  genuinely essential. So and do not conflict.

WHAT STILL KEEPS THE LETTER LOCANT (the live tripwire)
------------------------------------------------------
* a DIsubstituted urea — two substituents make the position essential (``:33327``);
* the CHALCOGEN analogues, even MONOsubstituted — §**** (``:33439``)
  "Chalcogen analogues of urea... **Preferred IUPAC names use the letter locants
  N, and N'.**", and its own worked PIN is the monosubstituted
  ``N-(butan-2-yl)selenourea (PIN)`` (``:33451``). thiourea/selenourea/tellurourea
  are one-orbit too, so ONLY the ``base_name == 'urea'`` scoping in
  ``composer._build_n_substituted_name`` keeps them from being over-stripped — that
  is what these rows guard.

The whole name is asserted (session a project rule): asserting "the numeral is absent"
would be satisfied by an abstention or a dropped substituent alike.
"""
import pytest

from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Unsubstituted parent — the retained PIN itself,:33308).
        ("NC(=O)N", "urea"),
        # ★ MONOsubstituted: the locant is OMITTED,:2943 `methylurea
        # (PIN)`). If a change ever re-introduces `N-methylurea`,:2943 has been
        # allowed to be overridden by a misreading of:33308.
        ("CNC(=O)N", "methylurea"),
        # ★ DIsubstituted -> the letter locants ARE essential and cited
        # 'N and N'';:33327). The mono-omission must not reach here.
        ("CNC(=O)NC", "N,N'-dimethylurea"),
        # Both substituents on ONE nitrogen -> unprimed pair, a different molecule from
        # the row above. Present so a "just drop the prime" change cannot pass.
        ("CN(C)C(=O)N", "N,N-dimethylurea"),
    ],
)
def test_urea_mono_omits_di_keeps_letter_locants(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected_pin,citation",
    [
        # ★ THE OVER-STRIP TRIPWIRE. The chalcogen analogues are one-orbit exactly
        # like urea, but (:33439) is a MORE SPECIFIC rule whose PIN
        # keeps the letter locant even when MONOsubstituted -- its own worked example
        # is `N-(butan-2-yl)selenourea (PIN)` (:33451). The `base_name == 'urea'`
        # scoping of the mono-omission is what protects these; if it is ever widened
        # to a structural orbit test, both of these collapse to `methylthiourea` /
        # `(propan-2-yl)selenourea` and this row catches it.
        ("CNC(=S)N", "N-methylthiourea", "P-66.1.6.1.3.1 :33439"),
        ("CC(C)NC(=[Se])N", "N-(propan-2-yl)selenourea", "P-66.1.6.1.3.1 :33451"),
    ],
)
def test_urea_chalcogen_analogues_keep_letter_locants(namer, smiles, expected_pin, citation):
    assert namer.name(smiles) == expected_pin
