"""Wave-0 RED unit suite for the quaternary-cation ``-aminium`` emitter (Phase
184 WS-E.1).

Written BEFORE the implementation (the Nyquist gate). The standalone quaternary
cation must emit the systematic ``-aminium`` SUFFIX PIN (D-04), NOT the non-PIN
trivial (``tetramethylammonium``/``choline``) and NOT ``''``. The assertions
MUST FAIL (not collection-error) on the pre-implementation HEAD. They flip to
GREEN as Plan 184-02 adds the ``CATION_QUATERNARY`` branch (N->C demote +
``find_principal_chain`` + ``-aminium`` suffix; D-05).

The two PROTECT assertions MUST stay byte-identical (D-06): the betaine
zwitterion ``(trimethylazaniumyl)acetate`` (azaniumyl PREFIX, P-74.1.3 — a
SEPARATE path that WS-E.1 must not touch). The phosphatidylcholine
``2-(trimethylazaniumyl)ethyl`` substituent path is gold-row-protected
(gold L677), not unit-tested here.

Every assertion docstring cites the verbatim Blue Book P-number.

Blue Book sources (BlueBookV2/BlueBookV2.md):
  - line 41354: ``(CH3)4N+ -> N,N,N-trimethylmethanaminium (PIN)``;
    ``tetramethylammonium``/``tetramethylazanium`` are explicitly non-PIN.
  - P-73.1.2.1 (lines 41429-41438), method (1) + Table 7.4: add ``ium`` to the
    amine suffix; the three other substituents are N-locant prefixes.
  - P-31: substituent (e.g. 2-hydroxy) locant numbering.
  - P-74.1.3: zwitterion azaniumyl prefix (the PROTECT path).

RED-by-design on HEAD (verified): ``C[N+](C)(C)C`` -> ``tetramethylammonium``,
``OCC[N+](C)(C)C`` -> ``choline``, ``CC[N+](C)(C)CC`` -> ``''``. PROTECT betaine
``C[N+](C)(C)CC(=O)[O-]`` -> ``(trimethylazaniumyl)acetate`` PASSES today.
"""

import pytest

from orthonym.namer import Orthonym


@pytest.mark.unit
class TestQuaternaryAminium:
    """WS-E.1 standalone quaternary cation -> systematic ``-aminium`` PIN (D-04).
    RED on HEAD by design (trivial name or '')."""

    def test_tetramethyl_quaternary(self):
        """P-73.1.2.1 (BlueBookV2 line 41354): (CH3)4N+ ->
        ``N,N,N-trimethylmethanaminium``; ``tetramethylammonium`` is non-PIN."""
        assert Orthonym().name("C[N+](C)(C)C") == "N,N,N-trimethylmethanaminium"

    def test_choline_quaternary(self):
        """P-73.1.2.1 + P-31: 2-hydroxyethyl-trimethyl quaternary cation ->
        ``2-hydroxy-N,N,N-trimethylethan-1-aminium`` (``choline`` is a non-PIN
        biological trivial). Accept the elided locant-1 variant."""
        assert Orthonym().name("OCC[N+](C)(C)C") in {
            "2-hydroxy-N,N,N-trimethylethan-1-aminium",
            "2-hydroxy-N,N,N-trimethylethanaminium",
        }

    def test_ethyldimethyl_quaternary(self):
        """P-73.1.2.1: N-substituent alphabetization (ethyl < methyl, di-
        ignored). CC[N+](C)(C)CC is N bonded to TWO ethyls + TWO methyls: parent
        = ethanaminium (one ethyl is the 2C chain through N), the other three
        branches = 1 ethyl + 2 methyl -> ``N-ethyl-N,N-dimethylethanaminium``.

        NOTE: the verbatim CONTEXT D-04 / RESEARCH A1 string
        ``N,N-diethyl-N-methylethanaminium`` does NOT round-trip to this SMILES
        (it parses to CC[N+](C)(CC)CC = three ethyls + one methyl). Corrected to
        the OPSIN-RT-confirmed PIN for the SMILES this phase actually targets
        (Rule 1 data fix; see 184-00 SUMMARY deviations)."""
        assert Orthonym().name("CC[N+](C)(C)CC") == "N-ethyl-N,N-dimethylethanaminium"


@pytest.mark.unit
class TestQuaternaryAminiumProtect:
    """D-06 PROTECT: the azaniumyl-PREFIX path (zwitterion / lipid substituent)
    is SEPARATE from the standalone aminium suffix and MUST stay byte-identical.
    GREEN on HEAD; must remain GREEN through WS-E.1."""

    def test_betaine_azaniumyl_prefix_unchanged(self):
        """P-74.1.3: the betaine zwitterion is named anion-is-parent with the
        cation as an ``azaniumyl`` PREFIX -> ``(trimethylazaniumyl)acetate``.
        WS-E.1 changes only the STANDALONE quaternary cation suffix path; this
        zwitterion route (charged_router._route_zwitterion / cation_to_prefix)
        is untouched.

        NOTE: the phosphatidylcholine ``2-(trimethylazaniumyl)ethyl`` substituent
        path is gold-row-protected (gold L677, LIPID@250), not unit-tested here.
        """
        assert Orthonym().name("C[N+](C)(C)CC(=O)[O-]") == "(trimethylazaniumyl)acetate"
