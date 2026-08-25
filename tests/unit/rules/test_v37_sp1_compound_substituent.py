"""v37 SP1.1 — compound-substituent enumeration: fail closed, never drop.

Root-cause class: an N-substituent fragment that carries a non-carbon heavy atom
(an ether O, a thioether S, a halogen, ...) was named purely by CARBON COUNT via
``naming_utils.get_alkyl_name(cc)`` inside ``composer._walk_amine_n_substituents``,
which SILENTLY DROPS the heteroatom. For ``COCCNCCC`` the N-substituent
``-CH2CH2-O-CH3`` (2-methoxyethyl) collapsed to ``propyl`` (O dropped, its two
flanking carbons walked as one 3-carbon chain), yielding the WRONG molecule
``N-propylpropan-1-amine`` (dipropylamine) — a different InChIKey that SELF-01
then suppressed to an abstain (breadth lost; a latent wrong-molecule producer).

The fix routes any N-substituent fragment carrying a non-carbon heavy atom to the
element-agnostic recursive namer (``name_substituent_fragment``) and FAILS CLOSED
(the whole amine build returns None) when it declines — it must never fall back to
the carbon-only ``get_alkyl_name`` that drops the heteroatom.

Verified fresh (`.venv/bin/python -m orthonym`) 2026-08-25; grounding:
 (lever 1), spy-confirmed on-path site
`composer._walk_amine_n_substituents` (the named leads substituent_enumerator /
composer:193/246/3947 were OFF-PATH, invariant 8).
"""
import pytest

from orthonym import name_compound
from orthonym import errors


# ---------------------------------------------------------------------------
# Producer-level (OPSIN validity gate OFF by default in this suite): proves the
# PRODUCER itself no longer emits the O-dropped wrong molecule — not merely that
# SELF-01 catches it downstream.
# ---------------------------------------------------------------------------
class TestProducerNoAtomDrop:
    def test_producer_no_longer_drops_ether_o(self):
        """Gate OFF: the raw producer must NOT emit the O-dropped dipropylamine.

        Today (pre-fix) the producer returns 'N-propylpropan-1-amine' (WRONG,
        the ether O dropped). After the fix it returns the correct PIN, which
        retains the methoxy on the ethyl arm."""
        name = name_compound('COCCNCCC')
        assert name != 'N-propylpropan-1-amine'          # never the O-dropped wrong molecule
        assert name == 'N-(2-methoxyethyl)propan-1-amine'


# ---------------------------------------------------------------------------
# Production config (OPSIN validity gate ON): the real deployment behaviour.
# ---------------------------------------------------------------------------
class TestMixedEtherAmineProduction:
    @pytest.mark.opsin_gate
    def test_mixed_ether_amine_n_substituent_no_atom_drop(self):
        # COCCNCCC = CH3-O-CH2CH2-NH-CH2CH2CH3 ; PIN N-(2-methoxyethyl)propan-1-amine
        # (UDZCEFCJEGGQOJ-UHFFFAOYSA-N)
        name = name_compound('COCCNCCC')
        assert name != 'N-propylpropan-1-amine'
        assert name == 'N-(2-methoxyethyl)propan-1-amine'


# ---------------------------------------------------------------------------
# The class: an element-agnostic set of amines whose N-substituent carries a
# non-carbon heavy atom (ether O, thioether S, halogen) and is the SUBSTITUENT
# (not the parent) — the exact shape that used to drop the heteroatom. Each
# abstained before the fix; each now names the RIGHT molecule (OPSIN-RT-exact,
# verified 2026-08-25). NEVER a wrong (atom-dropped) molecule.
# ---------------------------------------------------------------------------
class TestCompoundNSubstituentClass:
    @pytest.mark.parametrize("smi,expected", [
        ('COCCNCCC',    'N-(2-methoxyethyl)propan-1-amine'),        # ether O, propyl parent
        ('COCCNCCCC',   'N-(2-methoxyethyl)butan-1-amine'),         # ether O, butyl parent
        ('COCCCNCCCC',  'N-(3-methoxypropyl)butan-1-amine'),        # ether O, longer arm
        ('CSCCNCCCC',   'N-[2-(methylsulfanyl)ethyl]butan-1-amine'),  # thioether S (bracket escalation)
        ('ClCCNCCCC',   'N-(2-chloroethyl)butan-1-amine'),          # chloro
        ('FCCNCCCC',    'N-(2-fluoroethyl)butan-1-amine'),          # fluoro
        # tertiary + repeated compound N-substituent -> derived multiplier bis(...)
        ('COCCN(CCOC)CCCC', 'N,N-bis(2-methoxyethyl)butan-1-amine'),
        # tertiary, mixed compound + simple N-substituents (alpha order)
        ('COCCN(C)CCCC', 'N-(2-methoxyethyl)-N-methylbutan-1-amine'),
        # symmetric secondary bis-compound
        ('ClCCNCCCl',   '2-chloro-N-(2-chloroethyl)ethan-1-amine'),
    ])
    def test_hetero_n_substituent_named_or_abstains_never_wrong(self, smi, expected):
        name = name_compound(smi)
        # 0-wrong: either the RT-verified correct name, or a clean abstain —
        # never a partial (atom-dropped) wrong molecule.
        assert name == expected or errors.is_failure_name(name), (
            f"{smi}: got {name!r}, expected {expected!r} or a clean abstain")


# ---------------------------------------------------------------------------
# Controls: pure-hydrocarbon amines and hetero-arm-as-parent amines must be
# byte-identical (the fix only touches hetero-bearing N-substituent fragments).
# ---------------------------------------------------------------------------
class TestControlsUnchanged:
    @pytest.mark.parametrize("smi,expected", [
        ('CCNCCC', 'N-ethylpropan-1-amine'),         # pure-alkyl secondary amine
        ('CCNCC', 'N-ethylethanamine'),              # pure-alkyl secondary amine
        ('COCCNCC', 'N-ethyl-2-methoxyethan-1-amine'),   # hetero arm IS the parent
        ('COCCN(C)C', '2-methoxy-N,N-dimethylethan-1-amine'),  # tertiary, hetero=parent
        ('CCOCC', 'ethoxyethane'),                   # ether, no amine
    ])
    def test_control_unchanged(self, smi, expected):
        assert name_compound(smi) == expected
