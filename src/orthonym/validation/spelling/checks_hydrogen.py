"""Spelling check: indicated hydrogen (registered with
:func:`orthonym.validation.pin_spelling.register`).

The count of indicated hydrogens each ring system needs comes from the RDKit structure
(:mod:`.indicated_hydrogen`); the name side is its front ``nH-`` groups. No naming code, no OPSIN.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from . import indicated_hydrogen
from .checks_assembly import ASSEMBLY_HEAD
from .lexer import normalise


# ------------------------------------------------------------------ indicated hydrogen
@register('P-14.7.1')
def indicated_hydrogen_check(mol, name):
    """ (the Blue Book) "in a preferred IUPAC name a locant and the symbol 'H' must be
    cited"; (:24639). The name cites fewer front indicated-hydrogen groups than the
    ring systems of the structure need (:mod:`.indicated_hydrogen`). Charged ring atoms abstain
    (the ylium/uide parents are formed from the neutral parent hydride by operations the
    count does not model). A skeletal-replacement monocycle with explicit 'ene' endings names no
    mancude parent and needs none: (:16558) numbers its heteroatoms and then its
    unsaturated sites, and (heading:23692):23694 forms the PIN "by changing the 'ane'
    ending of the saturated heteromonocycle to 'ene', 'adiene', etc."
    ('1-azacyclotrideca-2,4,6,8,10,12-hexaene (PIN)':23698, not '1H-1-aza[13]annulene':16580)."""
    if re.search(r"a(?:cyclo)(?:prop|but|pent|hex|hept|oct|non|dec)a-\d+(?:,\d+)*-(?:di|tri|tetra)?en",
                 normalise(name)):
        return None
    if any(a.GetFormalCharge() and a.IsInRing() for a in mol.GetAtoms()):
        return None
    r = indicated_hydrogen.analyse(Chem.MolToSmiles(mol), normalise(name))
    if r.get('verdict') == 'OMISSION':
        return SpellingFailure('P-14.7.1', f"indicated hydrogen required {dict(r['reqd'])}, "
                                           f"cited {[g[1] for g in r['front']]}")
    return None


# ------------------------------------------------------------------ ring assemblies
#: an indicated-hydrogen group directly after the multiplying prefix of a ring assembly, inside the
#: parentheses of its component or after the hyphen of a locant-initial one: 'bi(4H-1,4-oxazine)',
#: 'bi-6H-pyran', 'bi(1H-indole)'
_IH_INSIDE_ASSEMBLY = re.compile(ASSEMBLY_HEAD + r"[(\[-]?\d+[a-z]?(?:,\d+[a-z]?)*H-")


@register('P-28.2.3')
def assembly_indicated_hydrogen_check(mol, name):
    """ (the Blue Book) "Any remaining saturated ring positions are designated as
    indicated hydrogen, placed together with the appropriate locant(s) at the front of the name of
    the assembly.";:15599 '6H,6'H-2,2'-bipyran (PIN) (not 2,2'-bi-6H-pyran)'; the next paragraph
    (:15595) calls the front position "a change from its position in previous editions (refs. 1
    and 2) where it was kept with the name of the individual ring, for example, 2,2'-bi-2H-pyran".
    A name that keeps the indicated hydrogen with the component, after the multiplying prefix
    ('4,4'-bi(4H-1,4-oxazine)'), is the previous editions' spelling, not the PIN. Lexical only."""
    if _IH_INSIDE_ASSEMBLY.search(normalise(name)):
        return SpellingFailure('P-28.2.3', "indicated hydrogen cited with the component after the "
                                           "assembly multiplier, not at the front of the assembly name")
    return None
