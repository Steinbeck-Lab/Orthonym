"""Every consumer of a doubly-bonded fragment takes the pipeline's morphology.

Phase 1b, consumer half.

Five namers cite a fragment that is DOUBLE-bonded to their core -- the
semicarbazone and hydrazone namers in ``composer``, the azine handler, the
cumulative ium/ide chain emitter in ``rules.ions``, and the P-64.5(3) ketene
branch. All five used to do the same thing:

    yl = name_substituent(mol, frag, c)
    if not yl.endswith("yl"):
        return None
    ... f"{yl}idene" ...

That is the free-valence morphology being decided a SECOND time, in the
consumer, by rewriting a token. It only ever worked because the pipeline was
handing back the wrong single-valence token for a double bond; the moment the
pipeline became correct, every one of these consumers rejected its own correct
input and fell through to a worse name or to none at all.

The fix is that the producer owns the morphology and the consumer verifies
rather than rewrites. These tests pin the end-to-end names so the surgery
cannot come back: each one FAILS both if the pipeline regresses to ``-yl`` and
if a consumer starts appending morphemes again.

References: IUPAC 2013 P-29.2; P-68.3.1.2.2 (hydrazone), P-68.3.1.2.3 (azine),
P-15.2.2 (semicarbazone), P-64.5(3) (ketene ylidene-methanone).
"""
import pytest

import orthonym

pytestmark = pytest.mark.unit


def _name(smiles):
    return orthonym.name_compound(smiles, style="pin")


class TestHydrazoneFamily:

    @pytest.mark.parametrize("smiles,expected", [
        ("CC=NN", "ethylidenehydrazine"),
        ("CCC=NN", "propylidenehydrazine"),
        ("CC(C)=NN", "(propan-2-ylidene)hydrazine"),
        ("C1CCCCC1=NN", "cyclohexylidenehydrazine"),
    ])
    def test_hydrazone_substitutive_pin(self, smiles, expected):
        assert _name(smiles) == expected

    def test_semicarbazone(self):
        assert _name("NC(=O)NN=C(C)C") == (
            "2-(propan-2-ylidene)hydrazine-1-carboxamide")


class TestAzine:

    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)=NN=C(C)C", "di(propan-2-ylidene)hydrazine"),
        ("CCC=NN=CCC", "dipropylidenehydrazine"),
        ("C1CCCCC1=NN=C1CCCCC1", "dicyclohexylidenehydrazine"),
    ])
    def test_symmetric_azine(self, smiles, expected):
        assert _name(smiles) == expected


class TestKeteneOxomethylidene:

    def test_ring_ylidene_methanone(self):
        """P-64.5(3): O=C=C(ring) -> '<ring-ylidene>methanone'."""
        assert _name("O=C=C1CCCCC1") == "cyclohexylidenemethanone"


class TestNoConsumerRewritesTheMorphology:
    """Source guard: the ``f"{yl}idene"`` pattern must not reappear.

    ``composer.py`` is deliberately NOT in this list. Its
    ``_convert_yl_to_ylidene`` is an older, separate mechanism on the ring
    substituent paths, and it is still load-bearing there for the
    ``get_alkyl_name(carbon_count)`` fallback, which counts carbons and can
    only produce a ``-yl`` form. It has been made idempotent and
    oracle-driven, so it no longer rewrites a token the pipeline already
    spelled correctly; retiring it outright belongs with retiring that
    fallback. What IS pinned for composer is behaviour, above.
    """

    SITES = [
        "src/orthonym/assembly/handlers/azine.py",
        "src/orthonym/rules/ions.py",
        "src/orthonym/rules/ketenes.py",
    ]

    @pytest.mark.parametrize("path", SITES)
    def test_no_idene_suffix_append(self, path):
        from pathlib import Path
        text = Path(path).read_text()
        for pattern in ('}idene', "+ 'idene'", '+ "idene"'):
            assert pattern not in text, (
                f"{path} rebuilds the P-29.2 morphology by appending "
                f"{pattern!r}; the substituent pipeline already emits it"
            )
