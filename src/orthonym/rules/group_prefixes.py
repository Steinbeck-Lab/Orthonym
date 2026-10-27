"""The prefix names that the substitutive path offers for one group of a molecule, best first.

No namer of a substituent group is right for every group: ``name_substituent_fragment`` names a group on a carbon
well and a group on a nitrogen by a carbon-count ('amino-N-methylmethyl' for dimethylamino) or drops a C=S
('(4-methoxyanilino)(sulfanyl)methyl' for the group of a thioamide); the ring-substituent chokepoint of the ligands
of a centre (``centre_ligands``) names a ring group and a ring '-ylidene' that the first one declines and spells the
marks of a compound prefix ('bis[...]') that the universal substituent namer of the best-effort tier ('di[...]') does
not; that namer names the rest in a systematic form. A producer that had its own order before (the azanide of two
groups: the first namer, the universal one, then the ligand one) keeps it (``ligand_first``). The producers of an ion that is named by its groups offer
each candidate to the read-back of the name they build (the full InChIKey of the input) and ship the first that
reads back; a name that reads back is the name of the input, so no candidate can ship a wrong structure.
"""
from itertools import product
from typing import Iterator, List, Optional, Sequence


def group_prefix_candidates(mol, atoms: Sequence[int], attach: int, free_valence_suffix: Optional[str] = None,
                            parent_atoms: Sequence[int] = (), ligand_first: bool = True) -> List[str]:
    """The distinct, non-empty prefix names of the group ``atoms`` attached to its parent through ``attach``,
    in the order of the namers above. ``free_valence_suffix`` ('yl', 'ylidene') keeps the names that end with it,
    'yl' not the '-ylidene' and '-ylidyne' ones."""
    from ..assembly.substituent_naming import name_substituent_fragment
    from ..metrics.provenance import best_effort_ctx
    from .centre_ligands import _ligand_prefix
    names = [name_substituent_fragment(mol, list(atoms), attach, list(parent_atoms))]
    if ligand_first:
        names.append(_ligand_prefix(mol, list(atoms), attach))
    if best_effort_ctx.get():
        from ..assembly.universal_substituent import name_universal_substituent_prefix
        names.append(name_universal_substituent_prefix(mol, list(atoms), attach))
    if not ligand_first:
        names.append(_ligand_prefix(mol, list(atoms), attach))
    out: List[str] = []
    for name in names:
        if not name or name in out:
            continue
        if free_valence_suffix == 'ylidene' and not name.endswith('ylidene'):
            name = _as_ylidene(name)
            if not name or name in out:
                continue
        if free_valence_suffix == 'yl' and name.endswith(('ylidene', 'ylidyne')):
            continue
        out.append(name)
    return out


def _as_ylidene(name: str) -> str:
    """The '-ylidene' of a group that a namer spelled '-yl' for want of the double bond to its parent, else ''.

    The universal namer names a ring group by its own atoms and counts the bond to the parent as a single one: the group
    '1,4-dihydroquinolin-4-yl' (C4 with one hydrogen) of a ring carbon that is double bonded to its parent has no
    hydrogen, and its hydro prefixes are those of the ylidene: "General methodology for naming substituent groups"
    (the Blue Book) forms '-ylidene' on the same stem and the same hydro prefixes, the free valence of two taking the
    two hydrogens that 'yl' leaves one of ('cyclohexa-2,5-dien-1-ylidene' beside 'cyclohexa-2,5-dien-1-yl'). The caller reads
    the name that it builds back to the full InChIKey, so a stem for which the form is another group is withheld there."""
    return name[:-2] + 'ylidene' if name.endswith('yl') else ''


def combinations(candidates: Sequence[Sequence[str]], limit: int = 12) -> Iterator[tuple]:
    """The choices of one candidate for each group, the first candidates first, at most ``limit`` of them."""
    for count, choice in enumerate(product(*candidates)):
        if count >= limit:
            return
        yield choice


def first_name_that_reads_back(mol, names, record: bool = True) -> str:
    """The first of ``names`` that reads back to the full InChIKey of ``mol`` (charges and protonation site
    included), the stereo of the input composed onto it when a group cites no descriptor of its own, and recorded
    as no PIN (``record``: a memoised producer records the name each time it hands it out, not when it makes it);
    '' when none does. ``names`` may be a generator: a candidate is built when it is asked for."""
    from ..metrics.provenance import record_non_pin_fragment
    from .charged_router import _cation_name_rt_ok, _stereo_composed
    for name in names:
        if not name:
            continue
        composed = name if _cation_name_rt_ok(name, mol) else _stereo_composed(name, mol)
        if composed and (composed == name or _cation_name_rt_ok(composed, mol)):
            if record:
                record_non_pin_fragment(composed)
            return composed
    return ''


def tier_key() -> tuple:
    """The tiers a call runs in (``centre_ligands._breadth_contexts``): the candidates of a group depend on them, so
    the memo of a producer that is asked twice for one molecule is keyed by them."""
    from .centre_ligands import _breadth_contexts
    return tuple(bool(ctx.get()) for ctx in _breadth_contexts())
