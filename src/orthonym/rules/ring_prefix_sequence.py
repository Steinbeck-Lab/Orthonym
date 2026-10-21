"""The prefix sequence of a ring substituent name: ``3-chloro-2-methyl`` of ``3-chloro-2-methylphenyl``.

Moved out of ``rules.ring_substituents.decorated_ring_substituent_name`` (the assembly block, unchanged
for every ring that carries no N-substituent) so that one home renders the sequence and an anilino
prefix can merge the N-substituent of its nitrogen into it.

 "Retained names" (the Blue Book): "The prefix name 'anilino' is retained as the
preferred prefix for C6H5-NH- with full substitution allowed." The substituents of the ring and of
the nitrogen are cited in ONE alphanumerical sequence, as they are in the name of the aniline
parent: "ALPHANUMERICAL ORDER" (the Blue Book), '2-bromo-4-chloro-N-(2,4-dibromophenyl)
aniline (PIN)' (:22250). An N-substituent that is also a substituent of the ring joins that group, the
locant N cited before the numbers: 'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)' (:32879).
"""
from __future__ import annotations

from typing import Dict, List, Optional


def ring_prefix_sequence(mol, groups: Dict[str, List[int]], atom_prefixes: Dict[int, List[str]],
                         order: Dict[int, int], attachment_atom: int,
                         n_prefix: Optional[str] = None) -> Optional[str]:
    """The detachable-prefix sequence of a ring substituent name, locants cited.

    ``groups`` maps a prefix name to the locants it takes in the chosen numbering, ``atom_prefixes``
    maps a ring atom to the prefix names it carries, ``order`` maps a locant to a ring atom and
    ``attachment_atom`` is the free valence. ``n_prefix`` is the name of the one substituent of the
    nitrogen of an anilino prefix (``'methyl'``): it is cited with the locant ``N`` inside the same
    sequence. ``None`` when a ring that takes no locants (``pentachloro``) also carries an
    N-substituent, a form this module does not define.
    """
    def _is_complex_prefix(nm: str) -> bool:
        # simple v1-table prefixes are bare lowercase words; a branched/stereo
        # substituent name carries a locant, hyphen, or enclosing mark.
        #: a COMPOUND substituent prefix (formed by
        # substitution, e.g. the alkoxycarbonyl family methoxycarbonyl/
        # ethoxycarbonyl/phenoxycarbonyl) is enclosed even though it is a bare
        # lowercase word -> '2-(methoxycarbonyl)cyclohexyl', not '2-methoxy...'.
        if nm.endswith('oxycarbonyl'):
            return True
        #: a SUBSTITUTED pnictogen-yl (diphenylstibanyl / dimethylarsanyl
        # /...bismuthanyl) is a compound prefix requiring enclosing marks, even
        # though it is a bare lowercase word; the bare parent-hydride-yl
        # ('arsanyl'/'stibanyl'/'bismuthanyl') stays simple (cf. naming_utils
        #.is_complex_substituent). Narrowly scoped so simple alkyl/aryl (methyl,
        # phenyl) keep the byte-identical legacy path.
        for _pn in ('arsanyl', 'stibanyl', 'bismuthanyl'):
            if nm.endswith(_pn) and nm != _pn:
                return True
        # The hyphen in the character class made this a copy of the compound
        # predicate; the carve-out is the shared primitive, so
        # 'tert-butyl' is simple (BB 16286 cites it bare) while
        # 'tert-butylsulfanyl' stays compound.
        from ..assembly.naming_utils import italicized_prefix_is_bare
        if italicized_prefix_is_bare(nm):
            return False
        # lever B: a compound FG-prefix + alkyl ('hydroxymethyl',
        # 'cyanomethyl', 'carboxymethyl', 'aminoethyl') is a SUBSTITUTED (compound)
        # substituent and takes enclosing marks even though it carries no
        # digit/hyphen/mark — the character-class heuristic below misses it, so a
        # ring decoration 'hydroxymethyl' was cited bare ('4-hydroxymethylphenyl').
        # BB PINs: '2-(hydroxymethyl)benzene-1,4-diol' (:6802), '(cyanomethyl)'
        # (:33087). Same set naming_utils.needs_brackets / is_complex_substituent
        # recognise; scoped to this class to keep every other decoration
        # byte-identical.
        from ..assembly.naming_utils import (
            _ALKYL_ROOTS_FULL as _ARF,
        )
        from ..assembly.naming_utils import (
            _COMPOUND_FG_PREFIXES as _CFG,
        )
        _nml = nm.lower()
        for _fg in _CFG:
            if _nml.startswith(_fg) and _nml[len(_fg):] in _ARF:
                return True
        # (the Blue Book): a compound prefix is enclosed, and the question has one
        # home, ``naming_utils.enclose_if_compound`` (the union of ``needs_brackets`` and
        # ``is_complex_substituent``). The family test above names only the functional prefixes of
        # ``_COMPOUND_FG_PREFIXES``, so a compound prefix built on 'methoxy', 'dimethylamino',
        # 'methylsulfanyl', 'phenoxymethyl'... was cited bare: '3-methoxymethyloxiran-2-yl' for
        # '3-(methoxymethyl)oxiran-2-yl'. The shared primitive has the last word.
        from ..assembly.naming_utils import enclose_if_compound as _eic
        if _eic(nm) != nm:
            return True
        return any(c in nm for c in '-()[]0123456789')

    # (the Blue Book): every substitutable position of the aromatic
    # ring other than the free valence carries the same substituent -> no locants
    # ('pentachlorophenoxy'; '1-chloro-2-(pentafluoroethyl)benzene (PIN)',:3023).
    _substitutable = sorted(
        p for p, i in order.items() if i != attachment_atom
        and (mol.GetAtomWithIdx(i).GetSymbol() == 'C' or atom_prefixes[i]
             or mol.GetAtomWithIdx(i).GetTotalNumHs() > 0))
    _complete = (len(groups) == 1 and len(_substitutable) >= 2
                 and all(mol.GetAtomWithIdx(i).GetIsAromatic()
                         and mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in order.values())
                 and sorted(next(iter(groups.values()))) == _substitutable)
    if _complete and n_prefix is not None:
        # the omission of locants of is not defined for a ring that also carries
        # an N-substituent: the caller keeps its own construction
        return None
    if _complete:
        from ..assembly.naming_utils import enclose_if_compound as _eic
        from ..assembly.naming_utils import multiplied_component as _mc
        (name, locs), = groups.items()
        prefix_str = _mc(len(locs), name, _eic(name))
    elif n_prefix is None and not any(_is_complex_prefix(nm) for nm in groups):
        # byte-identical legacy path — simple-prefix rings are unchanged, except that a
        # simple prefix with an italicized detachable part sorts by its Roman letters:
        # "ALPHANUMERICAL ORDER" (the Blue Book) "Nonitalic Roman letters are
        # considered first" (:3442); (:3495) lets italic letters decide only when
        # the Roman letters do not: '1-(butan-2-yl)-3-tert-butylbenzene (PIN)' (:3507), so
        # 'tert-butyl' is cited by 'butyl' (before 'hydroxy'), not by 't'.
        from ..assembly.naming_utils import (
            strip_italicized_structural_prefix as _strip_italic_prefix,
        )
        parts = []
        # only the italicized sec-/tert- part is set aside; every other name keeps the plain
        # string order it always had ('disulfanyl' stays under 'd')
        for name in sorted(groups, key=lambda nm: (_strip_italic_prefix(nm)[0], nm)):
            locs = sorted(groups[name])
            loc_str = ','.join(str(loc) for loc in locs)
            from ..assembly.naming_utils import multiplied_component as _mc
            parts.append(f'{loc_str}-{_mc(len(locs), name, name)}')
        prefix_str = '-'.join(parts)
    else:
        #: a complex (branched/stereo) prefix is enclosed; ordering is
        # by the alphanumerical key of its stereo-stripped base name:
        # '(1S)-1-chloroethyl' alphabetizes at 'c', descriptor + locant ignored).
        from ..assembly.naming_utils import (
            alpha_sort_key as _alpha_sort_key,
        )
        from ..assembly.naming_utils import (
            apply_enclosing_marks,
        )
        from .stereochemistry import strip_stereo as _strip_stereo

        def _sort_key(nm: str):
            return (_alpha_sort_key(_strip_stereo(nm)), nm)

        from ..assembly.naming_utils import _wrap_n_substituent, enclose_if_compound
        from ..assembly.naming_utils import multiplied_component as _mc

        parts = []
        # the N-substituent is one more entry of the sequence, or joins the ring group of the
        # same name ('N,4-dimethyl'); its locant N is cited before the numbers
        for name in sorted(set(groups) | ({n_prefix} if n_prefix else set()), key=_sort_key):
            labels = ['N'] * (name == n_prefix) + [str(loc) for loc in sorted(groups.get(name, ()))]
            if name in groups:
                rendered = apply_enclosing_marks(name, -1) if _is_complex_prefix(name) else name
            else:
                rendered = _wrap_n_substituent(enclose_if_compound(name))
            parts.append(f"{','.join(labels)}-{_mc(len(labels), name, rendered)}")
        prefix_str = '-'.join(parts)

    return prefix_str
