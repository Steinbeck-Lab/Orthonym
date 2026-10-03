"""PIN class program, Task 4: monocycles with double and triple bonds at the PIN tier.

 (the Blue Book): "The presence of one or more double or triple bonds in an
otherwise saturated parent hydride [except for parents with Hantzsch-Widman names...] is
denoted by changing the ending 'ane' of the name of a saturated parent hydride to 'ene' or
'yne'. Locants as low as possible are given to multiple bonds as a set, even though this may
at times give 'yne' endings lower locants than 'ene' endings. If a choice remains, preference
for low locants is given to the double bonds."

 (:16548): "In monocyclic homogeneous unsaturated compounds, one double or triple
bond is always allocated the locant '1'. When alone, the locant '1' is omitted in names."

 (:16558): "In rings modified by skeletal replacement ('a') nomenclature, low
locants are assigned first to heteroatoms and then to unsaturated sites."
'1,11-disilacycloicosa-5,7-dien-3-yne (PIN)' (:16570; "not 1,11-disilacycloicosa-4,6-dien-
8-yne; the locant set '3,5,7' is lower than '4,6,8'"), '1,10-disilacycloicosa-12,14,16-trien-
18-yne (PIN)' (:16576).

 (:21366): "For the endings 'ene' and 'yne' lower locants are assigned first to
the endings as a set without regard to type and then to 'ene' endings." 'cycloicosa-1,3-dien-
5-yne (PIN) > cycloicosa-1,7-dien-3-yne (PIN)' (:21372), '> cycloicosa-1,5-dien-3-yne (PIN)'
(:21374). (e)(ii) (:3290) puts the same criterion after the suffixes (c): "low locants
are given first to multiple bonds as a set and then to double bonds".

 (:21066): 'cycloicosyne (PIN)' (:21070), 'cycloicos-1-en-3-yne (PIN)' (:21074),
'1,2,5,6-tetrasilacyclooct-3-en-7-yne (PIN)' and '1,2,5,6-tetrasilacycloocta-3,7-diyne
(PIN)' (:21091): an eight-membered heteromonocycle with a triple bond takes the 'a' name, as
a Hantzsch-Widman parent has no 'yne' ending.

 (a) (:7595): the terminal 'e' of 'ene' is elided before 'yne':
'cyclopentadec-1-en-4-yne (PIN, ' (:7601). (:16497): "when the endings
'ene' and 'yne' are preceded by a multiplying prefix and a locant the letter 'a' is
inserted" ('cyclododeca-1,3,5,7,9-pentaen-11-yne (PIN)',:17179).

At the base both tiers built these molecules without the triple bond (the carbocycle path
classified a ring with a triple bond as saturated and numbered only the double bonds; the
cyclic 'a' builder declined every ring triple bond), so the default tier declined them and
best-effort shipped the general engine's names at systematic_verified.

Declined at the PIN tier (best-effort keeps its RT-exact name):
- a six-membered carbocycle with a triple bond: (:16904) names the hydro
  derivatives of benzene 'cyclohexene' and 'cyclohexadiene' only, and (:17173,
  :17175) names the didehydro derivative '1,2-didehydrobenzene (PIN)' ("cyclohexa-1,3-dien-
  5-yne (formerly called 'benzyne')"); the Blue Book gives no 'yne' PIN in this family;
- a double bond together with two or more triple bonds: (a) keeps the 'e' of
  'ene' before 'diyne' (it starts with 'd'), and does not say where the 'a'
  goes before a multiplied 'yne' that follows an 'ene'; the Blue Book has no such example.
"""
import pytest

from orthonym.assembly.composition_primitives import _build_hydrocarbon_name
from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("C1#CCCCCCCCCCCCCCCC=CC=C1", "cycloicosa-1,3-dien-5-yne"),                      #:21372
    ("C1#CCCC=CCCCCCCCCCCCCC=C1", "cycloicosa-1,7-dien-3-yne"),                      #:21372
    ("C1#CC=CCCCCCCCCCCCCCCC=C1", "cycloicosa-1,5-dien-3-yne"),                      #:21374
    ("C1#CCCCCCCCCCCCCCCCCC=C1", "cycloicos-1-en-3-yne"),                            #:21074
    ("C1#CC=CC=CC=CC=CC=C1", "cyclododeca-1,3,5,7,9-pentaen-11-yne"),                #:17179
    ("C1#CCCCCCCCCCCC=CC1", "cyclopentadec-1-en-4-yne"),                             #:7601
    ("C1#CC[SiH2]CCCCCCCCC[SiH2]CCC=CC=C1", "1,11-disilacycloicosa-5,7-dien-3-yne"),  #:16570
    ("C1#CC[SiH2]CCCCCCCC[SiH2]CC=CC=CC=C1", "1,10-disilacycloicosa-12,14,16-trien-18-yne"),  #:16576
    ("C1#C[SiH2][SiH2]C=C[SiH2][SiH2]1", "1,2,5,6-tetrasilacyclooct-3-en-7-yne"),     #:21091
    ("C1#C[SiH2][SiH2]C#C[SiH2][SiH2]1", "1,2,5,6-tetrasilacycloocta-3,7-diyne"),     #:21091
]

# Members of the class beyond the Blue Book rows, built by the same rules (each expected
# name is read back by OPSIN to the input's full InChIKey inside the helper).
CLASS_ROWS = [
    ("C1CCCCCCCCCCCCCCCCCC#C1", "cycloicosyne"),                    #:21070, 'when alone'
    ("C1CCCC#CCC1", "cyclooctyne"),
    ("CC1CCCCC#C1", "3-methylcyclohept-1-yne"),                     # cf. '3-bromocyclohex-1-ene (PIN)':3297
    ("OC1CCCCCC#C1", "cyclooct-2-yn-1-ol"),                         # (c) before (e)
    ("O=C1CCCCC#CC1", "cyclooct-3-yn-1-one"),
    ("OC(=O)C1CCCCCC#C1", "cyclooct-2-yne-1-carboxylic acid"),
    ("ClC1=CC#CCCCC1", "1-chlorocyclooct-1-en-3-yne"),             # (e) set {1,3}, then ene 1
    ("CC1=CCCCCCC#C1", "2-methylcyclonon-1-en-3-yne"),
    ("C1=CCCC#CCC1", "cyclooct-1-en-5-yne"),                        # set {1,5} tie, then ene 1
    ("C1CCC#CCCC#C1", "cyclonona-1,5-diyne"),
    ("C1#CC#CCCCCCCCCCCC1", "cyclopentadeca-1,3-diyne"),
    ("C1#CCCCCOCCCCC1", "1-oxacyclododec-6-yne"),                   #
    ("C1#CCOCCCC1", "1-oxacyclooct-3-yne"),                         # as the:21091 eight-membered ring
]

# Already the PIN at both tiers at the base (protection).
CONTROL_ROWS = [
    ("C1CC[SiH2]CCCCCCCCC[SiH2]CCC=CC=C1", "1,11-disilacycloicosa-4,6-diene"),
    ("C1CCCCCCCCCCCCCCCCCC=C1", "cycloicosene"),                    #:21070
    ("C1=CCCCCCC=CCCCCCCCCCCC1", "cycloicosa-1,8-diene"),           #:21074
    ("C1=CCCCCCC=C1", "cyclonona-1,3-diene"),
    ("C1=CC#CC=C1", "1,2-didehydrobenzene"),                        #:17175
]

DECLINED_ROWS = [
    "C1CCC#CC1",           # six-membered carbocycle with a triple bond,
    "C1#CC#CC=CCCCC1",     # one 'ene' with a multiplied 'yne' (a),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", CLASS_ROWS)
def test_class_member(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", DECLINED_ROWS)
def test_declined_at_the_pin_tier(smiles):
    assert_declined_at_default(smiles)


@pytest.mark.parametrize("stem,dbl,tpl,omittable,name", [
    # the ring bond locant is omitted only when the ring has ONE multiple bond
    ("cycloicos", [1], [3], True, "cycloicos-1-en-3-yne"),
    ("cyclopentadec", [1], [4], True, "cyclopentadec-1-en-4-yne"),
    ("cycloicos", [], [1], True, "cycloicosyne"),
    ("cycloicos", [1], [], True, "cycloicosene"),
    ("cyclohept", [], [1], False, "cyclohept-1-yne"),
])
def test_ring_bond_locant_spelling(stem, dbl, tpl, omittable, name):
    assert _build_hydrocarbon_name(stem, dbl, tpl, ring_bond_locant_omittable=omittable) == name
