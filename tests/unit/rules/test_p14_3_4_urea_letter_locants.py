"""The urea class keeps its LETTER locants — a P-14.3.4 tripwire, not a feature test.

v29 Phase C Task 7(a). This file exists so that a future widening of any P-14.3.4 licence
(L1/L3/L5/L6) cannot silently strip ``N-methylurea`` down to ``methylurea``.

WHY IT IS A TRIPWIRE AND NOT A DEFECT REPORT
--------------------------------------------
``BlueBookV2/BlueBookV2.md:2943`` prints, verbatim::

    CH3-NH-CO-NH2 methylurea (PIN)

and that row sits inside the example block of §**P-14.3.4.3** ("The locant is omitted in
monosubstituted symmetrical parent hydrides or parent compounds where there is only one kind
of substitutable hydrogen"). Read alone it says our ``N-methylurea`` is wrong.

It is outranked by three rule *sentences* plus a worked example of its own class:

* §**P-66.1.6.1.1.1** (``:33308``) — "The compound H2N-CO-NH2 has the retained name 'urea',
  which is the preferred IUPAC name, **with locants N and N'**, as shown above the structure
  below. The systematic name is 'carbonic diamide'. The locants 1, 2, and 3 have been used in
  the past and may be used in general nomenclature."
* ``:33318`` — "Numerical locants for urea are no longer used in the IUPAC preferred name."
* §**P-66.1.6.1.3.1** (``:33439``) — "Chalcogen analogues of urea are named by functional
  replacement nomenclature ... **Preferred IUPAC names use the letter locants N, and N'.**
  Numerical locants may be used for thiourea in general nomenclature."
* ``:1717``(c) — "Numerical locants are no longer used in IUPAC names for urea, thiourea,
  condensed ureas, semicarbazide, semicarbazone, and the cation uronium."
* and decisively, a **monosubstituted** member printed WITH its letter locant:
  ``:33451`` ``N-(butan-2-yl)selenourea (PIN)``.

So what the class rules withdraw is the **numerical** locant (1/2/3), never the letter one.
``methylurea`` at ``:2943`` is a lone example row in another rule's block. **Weigh rule
sentences above example rows when they disagree** — the same adjudication that this phase
already paid for once.

Mechanically, ``assembly/locant_omission.py::_has_letter_locant`` is what enforces this: any
non-plain-numeric locant makes ``scope_forces_locants`` return True, so no licence can reach
it. A predicate-level test for that already exists in
``tests/unit/assembly/test_locant_omission.py``. This file is the **end-to-end** half,
because a predicate-level assertion is green even when production never reaches the
predicate — the failure shape this project has now hit repeatedly.
"""
import pytest

from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Unsubstituted parent — the retained PIN itself (P-66.1.6.1.1.1, :33308).
        ("NC(=O)N", "urea"),
        # ★ THE TRIPWIRE. Monosubstituted: the letter locant survives, exactly as
        # :33451 'N-(butan-2-yl)selenourea (PIN)' survives. If a licence widening ever
        # turns this into 'methylurea', :2943 has been allowed to outrank :33308.
        ("CNC(=O)N", "N-methylurea"),
        # Both nitrogens substituted -> primed letter locant (P-66.1.6.1.1.1 'N and N''').
        ("CNC(=O)NC", "N,N'-dimethylurea"),
        # Both substituents on ONE nitrogen -> unprimed pair, a different molecule from
        # the row above. Present so a "just drop the prime" change cannot pass.
        ("CN(C)C(=O)N", "N,N-dimethylurea"),
    ],
)
def test_urea_keeps_letter_locants(namer, smiles, expected):
    """The WHOLE name is asserted, per session invariant 11.

    Asserting only "the numeral is absent" would be satisfied by an abstention
    ('unknown organic compound') and by a dropped substituent alike -- which is
    precisely why the whole string is pinned. (The two chalcogen siblings below
    DID abstain when this was written; they now name, and the xfail marker that
    recorded the gap has been retired.)
    """
    assert namer.name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected_pin,citation",
    [
        # Chalcogen analogues are in scope of P-66.1.6.1.3.1 (:33439) and take the same
        # letter locants. These ABSTAINED when this test was written (measured
        # 2026-07-29) and were recorded as a strict xfail so that the day either
        # became nameable the test would XPASS -> FAIL -> and force whoever built
        # it to confirm the SPELLING rather than discovering it years later.
        #
        # That day came: `bbec0a76` added the retained chalcogen-urea parent
        # (assembly/handlers/thiourea.py) and both rows became nameable. The
        # marker's own instruction has been carried out -- the letter locants
        # were verified surviving before it was removed:
        #
        #     CNC(=S)N        -> N-methylthiourea
        #     CC(C)NC(=[Se])N -> N-(propan-2-yl)selenourea
        #
        # Both carry the italic letter locant `N` and no numeral, as
        # P-66.1.6.1.3.1 (:33439) requires -- *"Preferred IUPAC names use the
        # letter locants N, and N'. Numerical locants may be used for thiourea in
        # general nomenclature."* -- restated at :33446 *"Numerical locants are no
        # longer used for thiourea in the IUPAC preferred name."*  The second row
        # is the Blue Book's OWN worked example for the rule, `N-(butan-2-yl)-
        # selenourea (PIN)` (:33451), in its propan-2-yl homologue.
        ("CNC(=S)N", "N-methylthiourea", "P-66.1.6.1.3.1 :33439"),
        ("CC(C)NC(=[Se])N", "N-(propan-2-yl)selenourea", "P-66.1.6.1.3.1 :33451"),
    ],
)
def test_urea_chalcogen_analogues_keep_letter_locants(namer, smiles, expected_pin, citation):
    assert namer.name(smiles) == expected_pin
