"""PIN class program, batch 2 end checks: N-substituted amides named by the fragment branch.

``assembly/composer.py:_assemble_amide_name`` names a chain amide whose molecule has a double
or triple bond from its fragments (acyl parent, suffix, prefixes, stereodescriptors). It put
the N-substituent prefix in front of that assembled acyl name, so the N-substituent was never
ordered with the acyl prefixes and a stereodescriptor set ended up behind it:
'N-ethyl-2-bromopent-4-ynamide', 'N-methyl-2-methylprop-2-enamide',
'N-propyl(2E)-4-chlorobut-2-enamide'. Task 8 (chain halogens kept when the chain is the
parent) added members with a double bond in the N-substituent:
'N-[2-(cyclohex-1-en-1-yl)ethyl]-2,2,2-trichloroacetamide' (milestone1500).

 (the Blue Book): "Simple prefixes (i.e., those describing atoms and
unsubstituted substituents) are arranged alphabetically; multiplicative prefixes, if
necessary, are then inserted and do not alter the alphabetical order already established."
 (:3477): "The name of a prefix for a substituent is considered to begin with the
first letter of its complete name." An N-substituent is one of the prefixes:
'2-hydroxy-N-methylpropanamide (PIN)',:32730), '3-chloro-N-(2-chlorophenyl)
naphthalene-2-sulfonamide (PIN)',:32881); one named like an acyl prefix joins
it in one multiplied group, 'N,N,2-trimethyl-3-{...}propanamide (PIN)',:21624).
 (:4826): "Stereodescriptors placed at the front of the complete name or name
fragment to which they apply".

Every PIN below reads back to the input's full InChIKey (OPSIN 2.9.0).
"""
import pytest

from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("C1CCC(=CC1)CCNC(=O)C(Cl)(Cl)Cl", "2,2,2-trichloro-N-[2-(cyclohex-1-en-1-yl)ethyl]acetamide"),
    ("C1CCC(=CC1)CCNC(=O)C(Br)Br", "2,2-dibromo-N-[2-(cyclohex-1-en-1-yl)ethyl]acetamide"),
    ("ClCCC(=O)NCCC1=CCCCC1", "3-chloro-N-[2-(cyclohex-1-en-1-yl)ethyl]propanamide"),
    ("C#CCC(Br)C(=O)NCC", "2-bromo-N-ethylpent-4-ynamide"),
    ("CC(=C)C(=O)NC", "N,2-dimethylprop-2-enamide"),
    ("CC(=C)C(=O)NCCC", "2-methyl-N-propylprop-2-enamide"),
    ("ClCC=CC(=O)NCCC", "4-chloro-N-propylbut-2-enamide"),
    ("ClC/C=C/C(=O)NCCC", "(2E)-4-chloro-N-propylbut-2-enamide"),
    ("C/C(Cl)=C/C(=O)NCCC", "(2Z)-3-chloro-N-propylbut-2-enamide"),
    ("C/C=C/C(=O)NC", "(2E)-N-methylbut-2-enamide"),
]

# The spellings the fragment branch shipped at pin_verified before the fix.
NOT_PIN_ROWS = [
    ("C1CCC(=CC1)CCNC(=O)C(Cl)(Cl)Cl", "N-[2-(cyclohex-1-en-1-yl)ethyl]-2,2,2-trichloroacetamide"),
    ("C1CCC(=CC1)CCNC(=O)C(Br)Br", "N-[2-(cyclohex-1-en-1-yl)ethyl]-2,2-dibromoacetamide"),
    ("ClCCC(=O)NCCC1=CCCCC1", "N-[2-(cyclohex-1-en-1-yl)ethyl]-3-chloropropanamide"),
    ("C#CCC(Br)C(=O)NCC", "N-ethyl-2-bromopent-4-ynamide"),
    ("CC(=C)C(=O)NC", "N-methyl-2-methylprop-2-enamide"),
    ("CC(=C)C(=O)NCCC", "N-propyl-2-methylprop-2-enamide"),
    ("ClCC=CC(=O)NCCC", "N-propyl-4-chlorobut-2-enamide"),
    ("ClC/C=C/C(=O)NCCC", "N-propyl(2E)-4-chlorobut-2-enamide"),
    ("C/C(Cl)=C/C(=O)NCCC", "N-propyl(2Z)-3-chlorobut-2-enamide"),
    ("C/C=C/C(=O)NC", "N-methyl(2E)-but-2-enamide"),
]

# Already the PIN before the fix (the N-substituent sorts first, or another producer).
CONTROL_ROWS = [
    ("C=C(C)C(=O)NCc1ccccc1", "N-benzyl-2-methylprop-2-enamide"),
    ("C=CC(=O)NC(C)C", "N-(propan-2-yl)prop-2-enamide"),
    ("OCC=CC(=O)N(C)C", "4-hydroxy-N,N-dimethylbut-2-enamide"),
    ("Clc1ccc(cc1)C(=O)NCCC1=CCCCC1", "4-chloro-N-[2-(cyclohex-1-en-1-yl)ethyl]benzamide"),
    ("ClCC(=O)NCCc1ccccc1", "2-chloro-N-(2-phenylethyl)acetamide"),
    ("C1CCCCC1CCNC(=O)C(Br)Br", "2,2-dibromo-N-(2-cyclohexylethyl)acetamide"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_not_pin_labelled(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# Batch 2 fix a performance pass (F-03): when the one-series name cannot be built, the N-prefix in front
# of the acyl name is kept but labelled below the PIN unless it is already the series order.
# The rendering failure is forced (the merge helper raises), on molecules no other test names,
# so no cached name can hide the path.
def _force_one_series_failure(monkeypatch):
    import orthonym.assembly.composer as composer
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("forced rendering failure")

    monkeypatch.setattr(composer, "_amide_with_identical_prefixes_merged", boom)
    return calls


def test_render_failure_fallback_is_not_labelled_pin(monkeypatch, caplog):
    from orthonym import Orthonym
    calls = _force_one_series_failure(monkeypatch)
    smiles = "ClC/C=C/C(=O)NCCCC"           # (2E)-N-butyl-4-chlorobut-2-enamide
    fallback = "N-butyl(2E)-4-chlorobut-2-enamide"
    with caplog.at_level("WARNING", logger="orthonym.assembly.composer"):
        res = Orthonym().name_tiered(smiles)
    assert calls, "the one-series builder was not reached"
    assert "amide_one_series_render_failed" in caplog.text
    assert not (res.get("name") == fallback and res.get("tier") == "pin_verified"), res
    assert res.get("tier") != "pin_verified", res


@pytest.mark.parametrize("fragments,n_subs,expected", [
    # no acyl prefix, no stereodescriptor: the N-prefix is the whole series
    ([("parent", "but"), ("suffix", "enamide")], [{"name": "methyl"}], True),
    # a stereodescriptor goes in front of the complete name,:4826)
    ([("stereo", "(2E)"), ("parent", "but")], [{"name": "methyl"}], False),
    # 'butyl' sorts before 'chloro',:3477): N-first is the series order
    ([("prefix", "chloro"), ("parent", "but")], [{"name": "butyl"}], True),
    # 'propyl' sorts after 'chloro': not the series order
    ([("prefix", "chloro"), ("parent", "but")], [{"name": "propyl"}], False),
    # one name on N and on the acyl chain is one multiplied group ('N,2-dimethyl')
    ([("prefix", "methyl"), ("parent", "prop")], [{"name": "methyl"}], False),
])
def test_n_prefix_first_series_order(fragments, n_subs, expected):
    from types import SimpleNamespace

    from orthonym.assembly.composer import _n_prefix_first_is_series_order
    frags = [SimpleNamespace(fragment_type=t, text=x) for t, x in fragments]
    assert _n_prefix_first_is_series_order(frags, n_subs) is expected
