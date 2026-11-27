"""Spelling check: a free acid group (carboxylic, sulfonic or sulfuric, phosphonic or phosphoric) cited as
a prefix under a carboxylate ester name (registered with:func:`orthonym.validation.pin_spelling.register`).

 (the Blue Book) ranks acids (7) above esters (9). "Esters cited as prefixes" (:31696):
"When... another group is present that has priority for citation as the principal group... an ester group is indicated
by prefixes as 'acyloxy'" ('3-(benzoyloxy)propanoic acid (PIN)', '2-(acetyloxy)ethane-1-sulfonic acid (PIN)');
"Partial esters of polybasic acids and their salts" (:31935) '2-chloro-6-(ethoxycarbonyl)benzoic acid (PIN)'. An ester
of an inorganic acid that keeps an acid OH is named by the method of 'hydrogen salts',:40993; 'methyl
dihydrogen phosphate (PIN)',:35940; 'methyl hydrogen sulfate (PIN)',:35968). So 'butyl 10-(sulfooxy)octadecanoate'
(PIN '1-butoxy-1-oxooctadecan-10-yl hydrogen sulfate') or '...{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl
dihexadecanoate' (PIN '2-aminoethyl... hydrogen phosphate') is correct but no PIN.

Not this group: a name with the word 'acid' or 'hydrogen'; an acid OH on a ring P or S; an ion, a radical, a second
fragment; the classes (nucleotides, sugars, amino acids and peptides: 'L-serine'); a carbonic half-ester (one group).
A mixed anhydride beside the free acid is caught too, rightly: "an acid is preferred to an anhydride",:36973;
'(acetyloxy)phosphonic acid (PIN)',:36999). Reads the structure and the name.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .models import P10

_FREE_ACID = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=O)[OX2H1]', '[SX4;!R](=O)(=O)[OX2H1]', '[PX4;!R](=O)[OX2H1]'))
_CARBOXYLATE_ESTER = Chem.MolFromSmarts('[CX3;!R](=O)[OX2;!R][#6]')
_CHARGE_PAIRS = tuple(Chem.MolFromSmarts(s) for s in ('[N+](=O)[O-]', '[#7]=[N+]=[N-]', '[#6]=[N+]=[N-]'))
_ACID_WORD = re.compile(r"(?<![a-z])(?:acid|(?:di|tri|tetra)?hydrogen)(?![a-z])")


@register('P-65.6.3.2.3')
def acid_beside_ester_check(mol, name):
    """A free acid group, senior to the esters, is cited as a prefix under a carboxylate ester name."""
    if not name or _ACID_WORD.search(name) or P10.search(name):
        return None
    # two groups: the acid centre is not the ester's carbonyl carbon (a carbonic half-ester R-O-CO-OH is one group)
    acid_centres = {m[0] for p in _FREE_ACID for m in mol.GetSubstructMatches(p)}
    ester_carbons = {m[0] for m in mol.GetSubstructMatches(_CARBOXYLATE_ESTER)}
    if not (acid_centres - ester_carbons) or not (ester_carbons - acid_centres):
        return None
    paired = {i for p in _CHARGE_PAIRS for m in mol.GetSubstructMatches(p) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() or (a.GetFormalCharge() and a.GetIdx() not in paired)
                                              for a in mol.GetAtoms()):
        return None
    return SpellingFailure('P-65.6.3.2.3', "a free acid group is senior to the esters and anhydrides (P-41): the name "
                                           "is that of the acid, or a 'hydrogen' ester of the inorganic acid, not an "
                                           "ester or anhydride name with the acid as a prefix")
