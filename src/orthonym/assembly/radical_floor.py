"""The floor's name of a molecule whose one radical centre is a skeletal heteroatom of a ring.

A ring Si, Ge or B atom that holds fewer bonds than its standard valence and carries the rest as radical electrons
(a silylene drawn ``[Si]`` in a ring, a ring ``[B]``) is the atom of its parent hydride with a lowered valence, which
the lambda convention writes: ``2λ2-sila...`` "Nonstandard bonding numbers", the Blue Book: "A nonstandard
bonding number of a neutral skeletal atom of a parent hydride is indicated by the symbol 'λn', cited in
conjunction with an appropriate locant"). The radical electrons are then already cited, and a '-ylidene' beside the lambda cites them a
second time: the floor's radical rung names the whole heavy-atom set as an '-ylidene' prefix with the centre as its
attachment atom, which gave ``spiro[2λ2-sila...]-2-ylidene``, a name OPSIN reads as another structure. The name of
the molecule itself, with the lambda convention and no ylidene, is the name of the species.

``radical_floor_names`` is the floor's candidates for any molecule with radical centres, for the radical offer
(``assembly/radical_offer.py``), the last resort of the best-effort tier: the names of the routes first, then the names of
the assembly of identical parent radicals (``assembly/radical_assembly.py``) and of the radical ion of a one-atom parent
(``assembly/radical_ion.py``).
"""
import contextlib
import logging
from typing import Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

#: skeletal heteroatoms whose lowered valence the floor writes with the lambda convention
_LAMBDA_ELEMENTS = frozenset({"Si", "Ge", "B"})


def skeletal_lambda_floor_name(mol, radical_site) -> Optional[str]:
    """The whole-molecule floor name of ``mol`` when its only radical centre is a ring Si, Ge or B atom whose
    radical electrons are its missing valence, else None."""
    atom = mol.GetAtomWithIdx(radical_site["atom_idx"])
    if (atom.GetSymbol() not in _LAMBDA_ELEMENTS or not atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetIsAromatic()):
        return None
    standard = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
    held = atom.GetTotalValence()
    if atom.GetNumRadicalElectrons() != radical_site["n_electrons"] or held + radical_site["n_electrons"] != standard:
        return None
    from .universal_substituent import name_universal_substitutive
    named = name_universal_substitutive(mol)
    name = getattr(named, "name", None)
    return name if name and "λ" in name else None


def radical_floor_names(mol):
    """The names of the floor for a molecule with radical centres, one at a time and in the order they are tried
    (a generator: the caller reads each back and stops at the first that reads back). Runs inside
    ``read_back_policy.radical_last_resort`` only (the radical offer, ``assembly/radical_offer.py``).

    The names of the book's spelling come first; when a writer gave a book spelling and none of the names reads back,
    the same names without the '<acyl>oxy' group of the shared acid engine follow, and then the names of the
    mechanical spelling, as the retry of the tier names a molecule again (``Orthonym._name_tiered_with_book_retry``)."""
    from .book_prefixes import mechanical_forms, watch_book_forms
    seen = set()
    with watch_book_forms() as fired:
        for name in _route_names(mol, True):
            seen.add(name)
            yield name
    if fired[0]:
        for acyloxy_leaf, mechanical in ((False, False), (False, True)):
            with mechanical_forms() if mechanical else contextlib.nullcontext():
                for name in _route_names(mol, acyloxy_leaf):
                    if name not in seen:
                        seen.add(name)
                        yield name
    # the names of the classes below come after every name the routes above build, so a molecule that one of those names
    # reads back for keeps it
    for name in _further_names(mol):
        if name not in seen:
            seen.add(name)
            yield name


def _further_names(mol):
    """The names that the routes do not build, once those have given every name they have:

    * the routes again with the stereo cascade of the floor reading the names of a radical (``read_back_policy.
      radical_floor_stereo``), for a molecule with stereo;
    * centres in separate identical parts of the molecule, an assembly of parent radicals, "ASSEMBLIES OF PARENT
      RADICALS" (the Blue Book): ``(<linker>-diyl)bis(<unit>)`` (``assembly/radical_assembly.py``);
    * one centre that is a charged atom, a radical ion, "RADICAL IONS DERIVED FROM PARENT HYDRIDES"
      (the Blue Book): ``<prefixes>boranuidylidyne`` (``assembly/radical_ion.py``)."""
    from ..perception.ions import get_radical_sites
    from ..validation.read_back_policy import radical_floor_stereo
    from .book_prefixes import mechanical_forms
    from .radical_assembly import radical_assembly_names
    from .radical_ion import radical_ion_names
    sites = get_radical_sites(mol)
    ion = len(sites) == 1 and mol.GetAtomWithIdx(sites[0]['atom_idx']).GetFormalCharge()
    stereo = (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
              or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()))
    try:
        if stereo:
            # the routes again with the stereo cascade of the floor reading the names of the radical (-r): the descriptors
            # of every branch, in the locants of the branch, where the plain names above carry none. The cascade
            # reads with -r only while the next name is built: the offer's own checks of a name read as they always read
            routes = _route_names(mol, True)
            while True:
                with radical_floor_stereo():
                    name = next(routes, None)
                if name is None:
                    break
                yield name
        if not ion and len(sites) < 2:
            return
        if ion:
            for mechanical in (False, True):    # the book's spelling of the substituents, then the mechanical one
                with mechanical_forms() if mechanical else contextlib.nullcontext():
                    for acyloxy_leaf in (True, False):
                        yield from radical_ion_names(mol, acyloxy_leaf=acyloxy_leaf)
            return
        # the book's spelling, then the mechanical one of the groups of the unit with the linker still in the book's
        # spelling of its ring, then the mechanical one of both
        for linker_book, mechanical_units in ((True, False), (True, True), (False, True)):
            for acyloxy_leaf in (True, False):
                yield from radical_assembly_names(
                    mol, book_forms=linker_book, acyloxy_leaf=acyloxy_leaf, mechanical_units=mechanical_units)
    except Exception:   # a class that fails to build a name has none: the names before it stand
        logger.debug("radical floor: the further classes gave up on %s", Chem.MolToSmiles(mol), exc_info=True)
        return


def _route_names(mol, acyloxy_leaf: bool):
    """The names of the routes of the floor in the spelling the caller is in, in the order they are tried.

    The centres that the lambda convention describes "The λ-convention", the Blue Book: a divalent or
    trivalent one) are cited inside the name; each other centre needs a free valence of the name itself. One such
    centre is the attachment atom of the prefix entry point (a single centre of any kind is that one: the
    '-ylidene' name of the prefix comes first, the lambda name of the whole molecule after it); several are carried
    by one parent "CHOICE OF PARENT RADICAL", the Blue Book, ``name_universal_polyradical``). A
    monovalent centre that none of these places has no citation, and the molecule keeps its abstention."""
    from ..perception.ions import get_radical_sites
    from .t4_coverage import _final_floor_with_ring_ylidene
    from .universal_substituent import (
        name_universal_polyradical, name_universal_substituent_prefix, radical_lambda_site_ids)
    sites = get_radical_sites(mol)
    lambda_ids = radical_lambda_site_ids(mol)
    attach_sites = list(sites) if len(sites) == 1 else [s for s in sites if s['atom_idx'] not in lambda_ids]
    if len(attach_sites) == 1 and attach_sites[0]['n_electrons'] in (1, 2, 3):
        site = attach_sites[0]
        if len(sites) == 1 and site['n_electrons'] >= 2:   # a monovalent ring centre has no lambda
            name = skeletal_lambda_floor_name(mol, site)
            if name:
                yield name
        all_heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        name = name_universal_substituent_prefix(
            mol, all_heavy, site['atom_idx'], bond_order=site['n_electrons'], cap_attachment=True,
            acyloxy_leaf=acyloxy_leaf)
        if name:
            yield name
    if not attach_sites or (len(sites) == 1 and lambda_ids):
        floor = _final_floor_with_ring_ylidene(mol)
        if floor is not None and floor.name:
            yield floor.name
    elif len(attach_sites) >= 2:
        for book in (True, False):  # the book's spelling of the substituents, then the mechanical one
            poly = name_universal_polyradical(mol, book_forms=book, acyloxy_leaf=acyloxy_leaf)
            if poly is not None and poly.name:
                yield poly.name
