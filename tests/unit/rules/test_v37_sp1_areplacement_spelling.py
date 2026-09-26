""" — / enclosing-marks fix for the mixed-acetal class.

`_name_ether_substituted_chain` assembled >=2 DISTINCT unlocanted alkoxy
prefixes with a bare ``''.join`` (``ethoxymethoxymethyl``); OPSIN then reads the
fused ``ethoxymethoxy`` as a single compound (chained) prefix = a DIFFERENT
constitution, so rejected the pretty substitutive name and the engine
shipped the ugly a-replacement rescue. The fix separates the co-cited distinct
prefixes with enclosing marks -> ``ethoxy(methoxy)methyl`` (OPSIN-RT-verified).

This is a best-effort SPELLING/quality fix: PIN-default ABSTAINS on these inputs
(no PIN to regress); 0-wrong is preserved by either way. Genuine chain
skeletal-replacement PINs (``2,5,8-trioxanonane``) must stay untouched.
"""
import os

os.environ.setdefault("ORTHONYM_JVM_BUDGET", "off")

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.substituent_naming import _name_ether_substituted_chain
from orthonym.validation.opsin_roundtrip import opsin_parse


def _sub(smiles):
    """Name the acyclic fragment of `smiles` as a substituent on its ring."""
    m = Chem.MolFromSmiles(smiles)
    ring = {a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic() or a.IsInRing()}
    frag = [a.GetIdx() for a in m.GetAtoms() if a.GetIdx() not in ring]
    attach = next(i for i in frag
                  if any(n.GetIdx() in ring for n in m.GetAtomWithIdx(i).GetNeighbors()))
    return _name_ether_substituted_chain(m, frag, attach, set())


def _be():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _ik(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m) if m else None


def _rt_ok(name, smi):
    """True iff OPSIN parses `name` back to the same constitution as `smi`."""
    parsed = opsin_parse(name)
    return bool(parsed) and _ik(parsed) == _ik(smi)


# --- FIX: >=2 DISTINCT unlocanted alkoxy prefixes get enclosing marks -
def test_mixed_acetal_encloses_later_distinct_prefix():
    assert _sub('c1ccccc1C(OC)OCC') == 'ethoxy(methoxy)methyl'
    assert _sub('c1ccccc1C(OCC)OCCC') == 'ethoxy(propoxy)methyl'


# --- COLLATERAL: single prefix + di-multiplier stay bare (no over-enclosure) ---
def test_single_and_identical_prefixes_stay_bare():
    assert _sub('c1ccccc1COC') == 'methoxymethyl'
    assert _sub('c1ccccc1COCC') == 'ethoxymethyl'
    assert _sub('c1ccccc1C(OC)OC') == 'dimethoxymethyl'


# --- End-to-end: the marked substitutive name now wins over the rescue + RTs ---
# opsin_gate: exercise the PRODUCTION validity gate (disabled suite-wide
# by default). Before the fix the unmarked `ethoxymethoxymethyl` is -
# suppressed and the a-replacement rescue ships (`oxa`/`cyclohexa-1,3,5-triene`);
# after the fix the marked name passes the gate and wins. Skips (never green-
# blind) if the OPSIN jar is absent.
@pytest.mark.opsin_gate
@pytest.mark.parametrize('smi', [
    'c1ccccc1C(OC)OCC',
    'C1CCCCC1C(OC)OCC',
    'c1ccccc1C(OCC)OCCC',
])
def test_besteffort_prefers_marked_substitutive_and_rts(smi, opsin_gate):
    out = _be().name(smi)
    # a-replacement rescue must NOT be what ships any more.
    assert 'oxa' not in out
    assert 'cyclohexa-1,3,5-triene' not in out
    # the marked substitutive substituent must be present...
    assert 'ethoxy(' in out
    #... and it must still round-trip (/ 0-wrong preserved).
    assert _rt_ok(out, smi), f"{out!r} did not round-trip to {smi}"


# --- COLLATERAL: genuine skeletal-replacement chain PINs unbroken -------------
def test_genuine_chain_areplacement_pins_unbroken():
    from orthonym import name_compound
    # fix a performance pass: (the Blue Book) needs four heterounits;
    # '(1) 1-methoxy-2-(2-methoxyethoxy)ethane (PIN)' (:27756), '(4) 2,5,8,11-
    # tetraoxadodecane (PIN)' (:27762). Was '2,5,8-trioxanonane'.
    assert name_compound('COCCOCCOC') == '1-methoxy-2-(2-methoxyethoxy)ethane'
    assert name_compound('COCCOCCOCCOC') == '2,5,8,11-tetraoxadodecane'
    assert name_compound('COCOCOCOC') == '2,4,6,8-tetraoxanonane'
