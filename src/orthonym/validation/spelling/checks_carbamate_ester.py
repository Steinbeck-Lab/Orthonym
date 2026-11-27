"""Spelling check: a carbamate ester named on a junior parent with the ester as an '(alkoxycarbonyl)' prefix
(registered with:func:`orthonym.validation.pin_spelling.register`).

 (the Blue Book): "All preferred IUPAC names for esters are named by functional class nomenclature";
 (:18182) ranks "9 Esters" above acid halides (10), amides and ureas (11), hydrazides, imides, nitriles,
aldehydes, ketones, alcohols, amines, imines and the classes of the heteroatom parents (21-40); '2-hydroxypropyl
(2-aminoethyl)carbamate (PIN)' (:30766). So a molecule whose senior class is one carbamate ester (an open O on carbon) is
named as the ester ('tert-butyl piperidine-1-carboxylate', 'benzyl phenylcarbamate', '2-imidodicarbonate' esters,:31049),
and a name on a ring, urea, sulfonyl halide or silane parent ('2-({[(9H-fluoren-9-yl)methoxy]carbonyl}amino)ethane-1-
sulfonyl chloride') is correct but no PIN.

Not this group: an acid (carboxylic, chalcogen, or of B, P, As, Sb, S, Se), an anhydride, another open ester (carboxylic or
carbonate on carbon, thio, noncarbon oxoacid, pseudoester incl. ring N), an ion, a radical, a second fragment; a cyclic
carbamate (ring O) or an oxime carbamate (O on N). Two or more carbamate esters are left to their own lead unless the name
has no ester word at all. Reads the structure and the name only.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .checks_polyol_ester import _on_one_parent

# the N of an azide or an isocyanate on the carbonyl makes another acid: 'carbonazidic acid (PIN)',:30834),
# 'ethyl carbonazidate', not a carbamate
_CARBAMATE_ESTER = Chem.MolFromSmarts('[#7;!$(N=[N+]=[N-]);!$(N=N#N);!$(N=C=[O,S,Se,Te])][CX3](=O)[OX2;!R][#6]')
_CHARGE_PAIRS = tuple(Chem.MolFromSmarts(s) for s in ('[N+](=O)[O-]', '[#7]=[N+]=[N-]', '[#6]=[N+]=[N-]'))
_SENIOR_OR_OTHER_ESTER = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S,Se,N])[OX2H1,SX2H1,SeX2H1]',               # carboxylic, chalcogen and imidic acids
    '[B,P,As,Sb,S,Se,Te,Si,N;!$([N+](=O)[O-])][OX2H1,SX2H1]',  # acids of noncarbon centres (boronic, arsonic,...)
    '[CX3](=[O,S])[OX2][OX2H1]', '[!#6;!#1](=O)[OX2][OX2H1]',   # peroxy acids
    '[CX3](=[O,S])[OX2,SX2][CX3]=[O,S]',                    # anhydrides
    '[!#6;!#1](=O)[OX2][!#6;!#1]=O', '[!#6;!#1](=O)[OX2][CX3]=[O,S]',   # anhydrides of noncarbon acids
    '[CX3;!$(C[#7])](=[O,S,Se])[OX2,SX2,SeX2;!R][#6]',       # other open esters on carbon (carboxylic, carbonate, thio)
    '[CX3;$(C[#7])](=S)[OX2,SX2;!R][#6]',                   # thiocarbamate esters
    '[CX3](=[NX2])[OX2,SX2;!R][#6]',                        # imidic esters ('ethyl...propanimidate')
    '[!#6;!#1;!#7](=O)[OX2;!R][#6]',                        # esters of noncarbon oxoacids
    '[CX3](=O)[OX2][Si,Ge,Sn,Pb,B,Al,Ga,In,Tl,P,As,Sb,Bi,$([#7;R;+0])]',   # pseudoesters
))
_ESTER_WORD = re.compile(r'(?:carbamate|carboxylate|imidodicarbonate)[)\]}]*$')


@register('P-65.6.3.2.1')
def carbamate_ester_name_check(mol, name):
    """A molecule whose senior class is a carbamate ester is named on another parent."""
    if not name or not mol.HasSubstructMatch(_CARBAMATE_ESTER):
        return None
    paired = {i for p in _CHARGE_PAIRS for m in mol.GetSubstructMatches(p) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() or (a.GetFormalCharge() and a.GetIdx() not in paired)
                                              for a in mol.GetAtoms()):
        return None
    if any(mol.HasSubstructMatch(p) for p in _SENIOR_OR_OTHER_ESTER):
        return None
    if _ESTER_WORD.search(name.strip()):
        return None
    return SpellingFailure('P-65.6.3.2.1', "a carbamate ester is the senior class (P-41 class 9); its functional-class "
                                           "ester name is the PIN, not a name with an '(alkoxycarbonyl)' prefix")


# ------------------------------------------------------------------ two or more carbamate esters
#: Two or more carbamate esters have a name of their own: a ring-N one is the ester of a ring carboxylic acid, senior to
#: the carbamic (carbonic) acid of an open N,:29886, 'pyrrolidine-1-carboxylic acid (PIN)'; 7a before
#: 7b,:18172/:18174), so the name ends in '-carboxylate'; open NH carbamates on one parent, or esters of one polyol, take
#: a multiplicative name: carbamic acid is a functional parent with substitutable hydrogen (d),:5166),
#: "Multiplicative nomenclature is preferred to substitutive nomenclature",:23178), and of its esters "Method
#: (1) leads to preferred IUPAC names" ('ethyl methyl 3,3'-oxydibenzoate (PIN)',,:31794-:31803;
#: 'propane-1,3-diyl bis(chloroacetate) (PIN)',,:31819). A name that cites one carbamate ester as a
#: prefix ('benzyl (4-{[(benzyloxy)carbonyl]amino}phenyl)carbamate', PIN 'dibenzyl 1,4-phenylenebis(carbamate)') is then
#: correct but no PIN.
_CARBAMATE_GROUP = Chem.MolFromSmarts('[#7;!$(N=[N+]=[N-]);!$(N=N#N);!$(N=C=[O,S,Se,Te])][CX3;!R](=O)[OX2;!R][#6]')
_S_CARBAMATE = Chem.MolFromSmarts('[#7][CX3](=O)[SX2][#6]')
_CARBAMATE_PREFIX = re.compile(r"oxy[\]\)\}]*carbonyl[\]\)\}]*amino|carbamoyl[\]\)\}]*oxy")


def _nh_carbon(mol, n_idx, c_idx):
    """The one carbon of an NH carbamate N (single bonds, not an acyl or imidoyl carbon), else None."""
    n = mol.GetAtomWithIdx(n_idx)
    if n.GetTotalNumHs() != 1 or n.GetDegree() != 2 or any(b.GetBondTypeAsDouble() != 1 for b in n.GetBonds()):
        return None
    other = [a for a in n.GetNeighbors() if a.GetIdx() != c_idx]
    if len(other) != 1 or other[0].GetAtomicNum() != 6:
        return None
    if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(other[0]).GetAtomicNum() in (7, 8, 16)
           for b in other[0].GetBonds()):
        return None
    return other[0].GetIdx()


@register('P-65.6.3.3.2.2.1')
def multi_carbamate_ester_name_check(mol, name):
    """Two or more carbamate esters named with one of them cited as a prefix, where a ring-N carboxylate or a
    multiplicative name exists."""
    if not name:
        return None
    groups = mol.GetSubstructMatches(_CARBAMATE_GROUP)
    if len({g[1] for g in groups}) < 2 or mol.HasSubstructMatch(_S_CARBAMATE):
        return None
    paired = {i for p in _CHARGE_PAIRS for m in mol.GetSubstructMatches(p) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() or a.GetIsotope()
                                              or (a.GetFormalCharge() and a.GetIdx() not in paired)
                                              for a in mol.GetAtoms()):
        return None
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()) or any(
            b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()):
        return None
    if any(mol.HasSubstructMatch(p) for p in _SENIOR_OR_OTHER_ESTER):
        return None
    end = name.strip()
    if any(mol.GetAtomWithIdx(g[0]).IsInRing() for g in groups):
        # (method (1) of one diol may cite the carbamate anion last: 'ethane-1,2-diyl azetidine-1-carboxylate
        # methylcarbamate',,:31831)
        if not re.search(r'carboxylate(?![a-z])', end):
            return SpellingFailure('P-65.1.2.2.2', "the ester of a ring-N carboxylic acid is senior to an open carbamate "
                                                   "(carbonic) ester: the name ends in '-carboxylate'")
        return None
    if not _CARBAMATE_PREFIX.search(end):
        return None
    n_carbons = [_nh_carbon(mol, g[0], g[1]) for g in groups]
    o_carbons = {g[4] for g in groups}      # (N, C, =O, O, the carbon on O)
    # unequal ester groups on unequal NH carbons: no multiplicative name says which sits where ('R R'
    # (4-methyl-1,3-phenylene)bis(carbamate)'); the book prints no such name, so the check stays out
    cut = Chem.FragmentOnBonds(mol, [mol.GetBondBetweenAtoms(g[3], g[4]).GetIdx() for g in groups],
                               dummyLabels=[(0, 0)] * len(groups))
    cut.UpdatePropertyCache(strict=False)
    frag_of = {a: f for f, atoms in enumerate(Chem.GetMolFrags(cut)) for a in atoms}
    esters = {Chem.MolFragmentToSmiles(cut, [a for a, f in frag_of.items() if f == frag_of[g[4]]]) for g in groups}
    rank = list(Chem.CanonicalRankAtoms(cut, breakTies=False))      # the core with the ester groups cut off
    equal = len(esters) == 1 or (None not in n_carbons and len({rank[c] for c in n_carbons}) == 1)
    if (equal and None not in n_carbons and len(set(n_carbons)) == len(n_carbons) and _on_one_parent(mol, n_carbons)) or (
            len(o_carbons) == len(groups) and _on_one_parent(mol, o_carbons)):
        return SpellingFailure('P-65.6.3.3.2.2.1', "two or more carbamate esters on one parent take a multiplicative "
                                                   "name ('dibenzyl 1,4-phenylenebis(carbamate)', 'propane-1,3-diyl "
                                                   "dicarbamate'); a name citing one of them as a prefix is no PIN")
    return None

