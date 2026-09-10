""" #44 : a stereodescriptor on an acyclic-alkyl substituent must
carry the stereocentre's substituent locant, not a bare ``(R)-``/``(S)-``, when
the centre is not the attachment atom.

 ("## **** NAMING OF STEREOISOMERS", ``the Blue Book``): a
substituent-group stereodescriptor is "preceded by a numerical or letter locant
to describe the position of the stereogenic unit when such locants are present".
The locant comes from the substituent's OWN principal-chain numbering (attach=1),
which ``_located_acyclic_alkyl_name`` derives from structure -- the
same numbering the constitutional name already uses. RT-invisible (OPSIN infers
the lone centre), so this is a pure spelling-layer conformance fix; 0-wrong holds.
"""
import pytest

from orthonym import name_compound


@pytest.mark.parametrize("smiles,expected", [
    # single stereocentre OFF the attachment atom -> located descriptor
    ("OC(=O)c1ccc(CC[C@@H](O)C)cc1", "4-[(3S)-3-hydroxybutyl]benzoic acid"),
    ("OC(=O)c1ccc(C[C@H](O)CO)cc1", "4-[(2S)-2,3-dihydroxypropyl]benzoic acid"),
    ("OC(=O)c1ccc(CCC[C@H](O)C)cc1", "4-[(4R)-4-hydroxypentyl]benzoic acid"),
    # stereocentre AT the attachment atom (loc == free-valence locant) -> unchanged
    ("OC(=O)c1ccc([C@@H](O)C)cc1", "4-[(1S)-1-hydroxyethyl]benzoic acid"),
])
def test_acyclic_alkyl_substituent_stereo_locant(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # located secondary-alkyl (stereocentre at attachment) must not regress
    ("CC(=O)N[C@@H](C)CC", "N-[(2S)-butan-2-yl]acetamide"),
    ("CC(=O)N[C@@H](C)CCC", "N-[(2S)-pentan-2-yl]acetamide"),
])
def test_attachment_stereocentre_unchanged(smiles, expected):
    assert name_compound(smiles) == expected
