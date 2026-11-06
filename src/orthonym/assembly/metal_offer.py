"""The other drawing of a metal drawing, offered beside the drawing as written.

Standard InChI ignores metal-ligand bonds, so ``Cl[Zn]Cl`` and ``[Cl-].[Cl-].[Zn+2]`` are one
compound with one standard InChIKey, and the producers name the second where they cannot name the
first.:func:`metal_disconnected_spelling_and_order` writes the ionic drawing of an input when the
scope table (``data/metal_scope.py``, mechanism ``DISCONNECT``) admits the metal, and
:func:`metal_connected_spelling_and_order` writes the bonded drawing of an organometallic cation
drawn beside its anion (``[Cl-].[Zn+]c1ccccc1`` gives ``Cl[Zn]c1ccccc1``; mechanism
``CONNECT_ANION``), the one the producers of organometallic names read.
:func:`make_metal_offer_scope` builds the decorator of ``Orthonym.name`` that names the drawing as
written and the other drawing (:func:`metal_rewrite_spelling_and_order`), each as an outermost
``name`` call of its own, and keeps the better result (:func:`name_with_metal_offer`). The name of
the other drawing is recorded as a non-PIN label (:func:`record_metal_rewrite_non_pin`).

RDKit gives a drawing with dative bonds no standard InChIKey, so no name can be read back against
it.:func:`metal_dative_split_spelling_and_order` writes the same drawing without those bonds
(mechanism ``DATIVE``); the key of that drawing is the reference key, an independent count of atoms
and charges ties it to the drawing given, and:func:`metal_offer_for` offers it, followed by its
own ionic drawing when ``DISCONNECT`` applies to it.

Kept out of ``namer.py`` so that file stays within its size budget; the namer imports these
names back."""
import contextvars
import functools
from typing import Callable, NamedTuple, Optional, Tuple

from rdkit import Chem

from .. import input_mol as _input_mol
from ..errors import OrthonymLimitError, is_failure_name

#: Set while:func:`name_with_metal_offer` names the other drawing of a metal drawing; read by
#::func:`record_metal_rewrite_non_pin`.
METAL_REWRITE_RUN = contextvars.ContextVar("orthonym_metal_rewrite_run", default=False)

#: Set for the whole of:func:`name_with_metal_offer`, so that the calls it makes name the strings
#: they are given and offer nothing of their own.
METAL_OFFER_OWNED = contextvars.ContextVar("orthonym_metal_offer_owned", default=False)

#: Set, with:data:`METAL_REWRITE_RUN`, while:func:`name_with_metal_offer` names a drawing that
#: derives from the dative split of the caller's drawing; read by
#::func:`record_metal_rewrite_non_pin`.
METAL_DATIVE_SPLIT_RUN = contextvars.ContextVar("orthonym_metal_dative_split_run", default=False)

#: The Blue Book rule a name reached through another drawing of a metal drawing breaks:
#: 'Structures for Lewis 'adducts'' (the Blue Book), sentence:37540 'One of the
#: fundamental principles of organic chemical nomenclature is to name the structure that is drawn'.
METAL_REWRITE_RULE = "P-68.1.6.1"


def record_metal_rewrite_non_pin(name) -> None:
    """The name of the other drawing of a metal drawing is recorded as a non-PIN label, so it is
    labelled below the PIN (systematic_verified) and the default tier declines it with the rule;
    the wider tiers keep it (the same molecule, named through a drawing that has the same
    standard InChIKey). The drawing as written is the one the name must describe
    , so the name of another drawing is never certified as its PIN, whatever label
    the other drawing's own name earns (``zinc dichloride`` for ``[Cl-].[Cl-].[Zn+2]``).

    A name of a drawing that derives from the dative split of the caller's drawing is also
    recorded as read back to that drawing, not to the drawing as written
    (``metrics.provenance.record_verified_through_split``): its row says ``opsin_split``."""
    if not METAL_REWRITE_RUN.get():
        return
    if not isinstance(name, str) or not name or is_failure_name(name):
        return
    from ..metrics.provenance import record_non_pin_label, record_verified_through_split
    split = METAL_DATIVE_SPLIT_RUN.get()
    record_non_pin_label(
        name, rule=METAL_REWRITE_RULE,
        detail=("the drawing as written has dative bonds and no standard InChIKey: it is named "
                "through its drawing without them, not as drawn" if split else
                "the drawing as written is named through another drawing of the same compound, "
                "not as drawn"))
    if split:
        record_verified_through_split(name)


def metal_disconnected_spelling_and_order(
        smiles: Optional[str]) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """(the string to name, the atom order) of the ionic drawing of ``smiles`` when a metal is
    joined to non-carbon atoms by single bonds and the scope table admits the metal to
    ``DISCONNECT`` (``data/metal_scope.py``), else None.

    The ionic drawing is the one a charge-conserving heterolytic disconnection gives: every
    single bond from the metal to an atom that is neither carbon nor a metal is broken, the metal
    gains +1 and the other atom -1 for each, and the atoms keep their hydrogens
    (``Cl[Zn]Cl`` gives ``[Cl-].[Cl-].[Zn+2]``). Standard InChI ignores metal-ligand bonds, so the
    two drawings of a compound have one standard InChIKey, and a drawing that the producers cannot
    name can be named through the one they can. The key does not bound the metal's charge, and it
    is blind to the position of a charge on an ambident anion, so the rewrite is accepted only
    when, beside the equal key,

    * the metal's charge after the disconnection is one of its ``valences`` in the table
      (:func:`perception.metals.single_bond_metal_heteroatom_sites`: ``Cl[Zn](Cl)N`` would give
      ``[Zn+3]`` with the same key and is left as it is);
    * an independent count of atoms and charges holds
      (:func:`perception.metals.atom_and_charge_ledger_ok`: element, isotope, hydrogens,
      radicals, the charge change declared by the broken bonds, the bond set and the net charge),
      on the molecule and again on the written string read back;
    * the string reads back as the molecule (equal canonical isomeric SMILES).

    A drawing with a metal-carbon bond, a multiple or dative bond from the metal, a bond direction
    or a chiral tag at a broken bond, a CXSMILES extension or a lone-pair stereocentre is not
    rewritten, and an ionic drawing has no bond to break, so a second application finds nothing.
    ``order[k]`` is the atom of RDKit's reading of ``smiles`` that atom ``k`` of the string stands
    for. Structures for Lewis 'adducts', "One of the fundamental principles of organic
    chemical nomenclature is to name the structure that is drawn": the caller offers the drawing
    as written as well (:func:`name_with_metal_offer`)."""
    return _metal_spelling_and_order(smiles, _DISCONNECT_KIND)


def metal_connected_spelling_and_order(
        smiles: Optional[str]) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """(the string to name, the atom order) of the bonded drawing of ``smiles`` when it holds an
    organometallic cation drawn beside the free anion that neutralises it and the scope table
    admits the metal to ``CONNECT_ANION`` (``data/metal_scope.py``), else None.

    The bonded drawing joins the donor atom of the anion to the metal by a single bond; the metal
    and the donor each lose one charge (``[Cl-].[Zn+]c1ccccc1`` gives ``Cl[Zn]c1ccccc1``) and the
    atoms keep their hydrogens. It is the reverse of:func:`metal_disconnected_spelling_and_order`
    for the bond to the anion, and the bond to carbon stays: the standard InChIKey of the two
    drawings is equal (VERIFIED for an alkyl, an aryl and a benzyl cation), which is what lets the
    producers of organometallic names, which read a bonded drawing, name the drawing as written.
    The rewrite is accepted only under the conditions of
    :func:`perception.metals.connect_anion_to_organometal_cation_sites` (one cation, as many free
    anions as it has charges, no choice of pairing, an allowed metal valence), the equal key, the
    count of atoms and charges of:func:`perception.metals.atom_and_charge_ledger_ok` read the other
    way (the bonded drawing with its new bonds broken is the molecule), on the molecule and again on
    the written string read back, and the equal canonical isomeric SMILES of that read-back.

    A bonded drawing has no free anion to connect, so a second application finds nothing. A drawing
    of a carbanion beside a halometal cation (``[CH2-]C.[Zn+]Br``) is not a drawing of this
    kind: the carbon of the carbanion is no donor, and the bonded drawing that joins it has another
    key. Structures for Lewis 'adducts', "One of the fundamental principles of organic
    chemical nomenclature is to name the structure that is drawn": the caller offers the drawing
    as written as well (:func:`name_with_metal_offer`)."""
    return _metal_spelling_and_order(smiles, _CONNECT_KIND)


def metal_rewrite_spelling_and_order(
        smiles: Optional[str]) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """The other drawing the offer makes beside the drawing as written: the ionic drawing
    (:func:`metal_disconnected_spelling_and_order`) or, when that is none, the bonded drawing of an
    organometallic cation with its anion (:func:`metal_connected_spelling_and_order`). The two
    never both apply: the first refuses a drawing with a bond from a metal to carbon, the second
    needs one."""
    return metal_disconnected_spelling_and_order(smiles) or metal_connected_spelling_and_order(smiles)


def metal_dative_split_spelling_and_order(
        smiles: Optional[str]) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """(the string to name, the atom order) of the drawing of ``smiles`` without its dative bonds
    when every dative bond of it ends at a metal that the scope table admits to ``DATIVE``
    (``data/metal_scope.py``), else None.

    The split drawing keeps everything but the bonds: the donor its lone pair and its hydrogens,
    the metal its charge (``CC(=O)[O-]->[Zn+2]<-[O-]C(C)=O`` gives
    ``CC(=O)[O-].[Zn+2].[O-]C(C)=O``). RDKit gives a drawing with dative bonds no standard
    InChIKey, so the key of the split drawing is the reference key of the read-back of a name, and
    the rewrite is accepted only when it has one and when, beside it,

    * the sites are those of:func:`perception.metals.dative_bond_split_sites` (a metal admitted
      to ``DATIVE``, no stereo information at the bonds);
    * an independent count of atoms and charges holds
      (:func:`perception.metals.atom_and_charge_ledger_ok` with no charge moved: element, isotope,
      hydrogens, radicals, the charge of each atom, the bond set without the dative bonds and the
      net charge), on the molecule and again on the written string read back;
    * the string reads back as the molecule (equal canonical isomeric SMILES).

    The proof is of the atoms, the charges and the organic part; the dative bonds themselves are
    not compared, which is the strength of the proof for a drawing with covalent metal-ligand
    bonds as well (standard InChI ignores those bonds too). A drawing that holds a dative bond
    that no metal accepts is not split, and a split drawing has no dative bond, so a second
    application finds nothing. ``order[k]`` is the atom of RDKit's reading of ``smiles`` that atom
    ``k`` of the string stands for. Structures for Lewis 'adducts' draws the 'dative'
    bond as structure (I) and says "One of the fundamental principles of organic chemical
    nomenclature is to name the structure that is drawn": a name reached through the split drawing
    is not the name of that structure (:func:`record_metal_rewrite_non_pin`)."""
    return _metal_spelling_and_order(smiles, _DATIVE_KIND)


def metal_radical_joined_spelling_and_order(
        smiles: Optional[str]) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """(the string to name, the atom order) of the bonded drawing of ``smiles`` when its one metal atom has no bond
    and its carbon radicals are the carbons that lost their bond to it, and the scope table admits the metal to
    ``RECONNECT_RADICAL`` (``data/metal_scope.py``), else None.

    The bonded drawing joins each carbon radical to the metal by a single bond that takes the carbon's radical
    electron; no charge moves and the atoms keep their hydrogens (``[Li].CCC[CH2]`` gives ``CCCC[Li]``,
    ``[CH]1CC1.[Br-].[Mg+]`` gives ``[Mg+]C1CC1.[Br-]``). Standard InChI ignores metal-ligand bonds and radicals, so
    the two drawings have one standard InChIKey, and the rewrite is accepted only when, beside the equal key,

    * the sites are those of:func:`perception.metals.radical_join_sites` (carbons with one radical electron
      each, the metal's charge plus its new bonds one of its ``valences``);
    * an independent count of atoms, charges and radical electrons holds
      (:func:`perception.metals.atom_and_charge_ledger_ok` with ``moved_radical``: each new bond takes one radical
      electron from its carbon, nothing else changes), on the molecule and again on the written string read back;
    * the string reads back as the molecule (equal canonical isomeric SMILES).

    A bonded drawing has no radical carbon beside a bond-free metal, so a second application finds nothing.
     Structures for Lewis 'adducts', "One of the fundamental principles of organic chemical nomenclature
    is to name the structure that is drawn": the caller offers the drawing as written as well
    (:func:`name_with_metal_offer`), and the name of the bonded drawing is labelled below the PIN
    (:func:`record_metal_rewrite_non_pin`)."""
    return _metal_spelling_and_order(smiles, _RADICAL_KIND)


class MetalOffer(NamedTuple):
    """The drawings the offer names beside the drawing as written: ``drawings`` are
    ``(string, atom order)`` pairs, the nearest to the caller's drawing first, the order being
    relative to RDKit's reading of the caller's string; ``keyless`` is True for a drawing that has
    no standard InChIKey (dative bonds)."""
    drawings: Tuple[Tuple[str, Tuple[int, ...]], ...]
    keyless: bool = False


def metal_offer_for(smiles: Optional[str]) -> Optional[MetalOffer]:
    """The drawings:func:`name_with_metal_offer` names beside the drawing ``smiles`` as written,
    or None when there is none.

    A dative drawing is offered as its split drawing
    (:func:`metal_dative_split_spelling_and_order`) and, when that drawing has an ionic drawing of
    its own (:func:`metal_rewrite_spelling_and_order`: the split drawing is treated as any
    drawing), that one as well; the order of the second is composed to the caller's atoms. A drawing of carbon
    radicals beside a bond-free metal is offered its bonded drawing
    (:func:`metal_radical_joined_spelling_and_order`) and, in the same way, that drawing's own other drawing
    (``[Mg+]C1CC1.[Br-]`` its ``CONNECT_ANION`` drawing). Any other drawing is offered its other drawing, as before."""
    split = metal_dative_split_spelling_and_order(smiles)
    if split is not None:
        written, order = split
        again = metal_rewrite_spelling_and_order(written)
        if again is None:
            return MetalOffer((split,), keyless=True)
        return MetalOffer((split, (again[0], tuple(order[k] for k in again[1]))), keyless=True)
    joined = metal_radical_joined_spelling_and_order(smiles)
    if joined is not None:
        written, order = joined
        again = metal_rewrite_spelling_and_order(written)
        if again is None:
            return MetalOffer((joined,))
        return MetalOffer((joined, (again[0], tuple(order[k] for k in again[1]))))
    other = metal_rewrite_spelling_and_order(smiles)
    return None if other is None else MetalOffer((other,))


def _metal_spelling_and_order(smiles, kind):
    """The reading shared by the rewrites: the molecule, the sites that ``kind`` (a
    :class:`_RewriteKind`) finds in it and the checked spelling of the rewritten drawing."""
    if not smiles or '|' in smiles:
        return None
    try:
        from rdkit import rdBase
        from ..namer import _has_lone_pair_stereocentre
        # (the engine reads the string itself and logs what is wrong with it, once: this reading
        # is silent. InChI warns on every metal drawing, "Metal was disconnected": the comparison
        # of the keys below is what they are for, so those warnings are not logged either)
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            sites = kind.find_sites(mol)
            if not sites or _has_lone_pair_stereocentre(smiles):
                return None
            return _metal_spelling_checked(mol, sites, kind)
    except Exception:
        return None


def _hold_hydrogens(rw, atom_indices):
    """Fix the hydrogen count of the atoms, so that a change of charge or of bonds adds none."""
    for idx in atom_indices:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)


def _break_bonds(mol, sites):
    """``mol`` with the bond of each site ``(metal, other atom)`` broken: the metal gains +1 and
    the other atom -1, and the atoms keep their hydrogens. Sanitized, same atom order."""
    rw = Chem.RWMol(mol)
    _hold_hydrogens(rw, {other_idx for _metal, other_idx in sites})
    for metal_idx, other_idx in sites:
        rw.RemoveBond(metal_idx, other_idx)
        metal, other = rw.GetAtomWithIdx(metal_idx), rw.GetAtomWithIdx(other_idx)
        metal.SetFormalCharge(metal.GetFormalCharge() + 1)
        other.SetFormalCharge(other.GetFormalCharge() - 1)
    new = rw.GetMol()
    Chem.SanitizeMol(new)
    return new


def _join_bonds(mol, sites):
    """``mol`` with a single bond between the two atoms of each site ``(metal, donor atom)``: the
    metal loses one positive charge, the donor one negative charge, and the atoms keep their
    hydrogens. Sanitized, same atom order."""
    rw = Chem.RWMol(mol)
    _hold_hydrogens(rw, {donor_idx for _metal, donor_idx in sites})
    for metal_idx, donor_idx in sites:
        rw.AddBond(metal_idx, donor_idx, Chem.BondType.SINGLE)
        metal, donor = rw.GetAtomWithIdx(metal_idx), rw.GetAtomWithIdx(donor_idx)
        metal.SetFormalCharge(metal.GetFormalCharge() - 1)
        donor.SetFormalCharge(donor.GetFormalCharge() + 1)
    new = rw.GetMol()
    Chem.SanitizeMol(new)
    return new


def _remove_dative_bonds(mol, sites):
    """``mol`` without the dative bond of each site ``(metal, donor atom)``: no charge moves, and
    the atoms keep their hydrogens. Sanitized, same atom order."""
    rw = Chem.RWMol(mol)
    _hold_hydrogens(rw, {idx for site in sites for idx in site})
    for metal_idx, donor_idx in sites:
        rw.RemoveBond(metal_idx, donor_idx)
    new = rw.GetMol()
    Chem.SanitizeMol(new)
    return new


def _find_dative_sites(mol):
    from ..perception.metals import dative_bond_split_sites
    return dative_bond_split_sites(mol)


def _find_disconnect_sites(mol):
    from ..perception.metals import single_bond_metal_heteroatom_sites
    return single_bond_metal_heteroatom_sites(mol)


def _find_connect_sites(mol):
    from ..perception.metals import connect_anion_to_organometal_cation_sites
    return connect_anion_to_organometal_cation_sites(mol)


def _find_radical_sites(mol):
    from ..perception.metals import radical_join_sites
    return radical_join_sites(mol)


def _join_radical_bonds(mol, sites):
    """``mol`` with a single bond between the two atoms of each site ``(metal, carbon radical)``: the bond takes
    the carbon's radical electron, no charge moves, the atoms keep their hydrogens, and the metal carries no
    radical electron. Sanitized, same atom order."""
    rw = Chem.RWMol(mol)
    _hold_hydrogens(rw, {idx for site in sites for idx in site})
    for metal_idx, carbon_idx in sites:
        rw.AddBond(metal_idx, carbon_idx, Chem.BondType.SINGLE)
        carbon = rw.GetAtomWithIdx(carbon_idx)
        carbon.SetNumRadicalElectrons(carbon.GetNumRadicalElectrons() - 1)
    for metal_idx in {metal_idx for metal_idx, _carbon in sites}:
        rw.GetAtomWithIdx(metal_idx).SetNumRadicalElectrons(0)
    new = rw.GetMol()
    Chem.SanitizeMol(new)
    return new


class _RewriteKind(NamedTuple):
    """One rewrite: the sites of the molecule it acts on, the rewritten molecule, whether it forms
    the bonds of the sites (``CONNECT_ANION``) or removes them, the charge each bond of the sites
    carries in the count of atoms and charges (1 for a bond that is heterolysed or joined, 0 for
    a dative bond), whether the molecule has no standard InChIKey of its own (the key of the
    rewritten drawing is then the reference key), and whether each bond of the sites carries a radical
    electron instead (``RECONNECT_RADICAL``)."""
    find_sites: Callable
    rewrite: Callable
    bonds_are_formed: bool
    moved_charge: int = 1
    keyless_input: bool = False
    moved_radical: bool = False


_DISCONNECT_KIND = _RewriteKind(_find_disconnect_sites, _break_bonds, False)
_CONNECT_KIND = _RewriteKind(_find_connect_sites, _join_bonds, True)
_DATIVE_KIND = _RewriteKind(_find_dative_sites, _remove_dative_bonds, False, 0, True)
_RADICAL_KIND = _RewriteKind(_find_radical_sites, _join_radical_bonds, True, 0, False, True)


def _metal_spelling_checked(mol, sites, kind) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """The body of the rewrites once ``sites`` are found: the rewrite, the equal-key check (the
    key of the rewritten drawing is the reference when the molecule has none), the count of atoms
    and charges, the string and its read-back.

    The count of atoms and charges (:func:`perception.metals.atom_and_charge_ledger_ok`) takes the
    molecule that has the bonds first: the molecule when the rewrite removes them, the rewritten
    drawing when it forms them."""
    try:
        from ..namer import _smiles_output_order_of
        from ..perception.metals import atom_and_charge_ledger_ok

        def ledger_ok(rewritten):
            if kind.bonds_are_formed:
                return atom_and_charge_ledger_ok(rewritten, mol, sites, kind.moved_charge, kind.moved_radical)
            return atom_and_charge_ledger_ok(mol, rewritten, sites, kind.moved_charge, kind.moved_radical)

        if kind.keyless_input:
            new = kind.rewrite(mol, sites)
            key = Chem.MolToInchiKey(new)
            if not key or not ledger_ok(new):
                return None
        else:
            key = Chem.MolToInchiKey(mol)
            if not key:
                return None
            new = kind.rewrite(mol, sites)
            if Chem.MolToInchiKey(new) != key or not ledger_ok(new):
                return None
        written = Chem.MolToSmiles(new, canonical=False)
        order = _smiles_output_order_of(new)
        count = mol.GetNumAtoms()
        if order is None or sorted(order) != list(range(count)):
            return None
        back = Chem.MolFromSmiles(written)
        if (back is None or back.GetNumAtoms() != count or Chem.MolToInchiKey(back) != key
                or Chem.MolToSmiles(back) != Chem.MolToSmiles(new)):
            return None
        inverse = [0] * count
        for position, atom_idx in enumerate(order):
            inverse[atom_idx] = position
        if not ledger_ok(Chem.RenumberAtoms(back, inverse)):
            return None
        return written, tuple(order)
    except Exception:
        return None


def form_result_rank(namer, string, result):
    """The rank of the result of an outermost ``name`` run of ``string`` by ``namer``, read right
    after the run (its provenance and its default-tier row are this run's): ``(the label rank, 0)``
    for a name (``namer._LABEL_RANK``; a name whose row cannot be derived ranks 1), ``(0, 1)`` for
    a decline of the default tier that says a name exists below the PIN, ``(0, 0)`` for any
    other failure label: of two runs that name nothing at the default tier, the one whose decline
    is NO_VERIFIED_PIN (a wider tier keeps its name) stands over one that names nothing."""
    from ..metrics import provenance as _pv
    from ..namer import _LABEL_RANK
    if not isinstance(result, str) or not result or is_failure_name(result):
        stash = getattr(namer, "_default_tier_row", None)
        declined = (isinstance(stash, tuple) and len(stash) > 1
                    and isinstance(stash[1], dict) and stash[0] == result
                    and stash[1].get("limit_code") == "NO_VERIFIED_PIN")
        return (0, 1 if declined else 0)
    try:
        row, _pin_form = namer._tier_row_and_pin_form(string, result, _pv.get_provenance())
        return (_LABEL_RANK.get(row.get("tier"), 0), 0)
    except Exception:
        return (1, 0)


def name_with_metal_offer(namer, fn, offer, args, kwargs) -> str:
    """The drawing of the caller's input as written and the other drawings of ``offer``
    (:class:`MetalOffer`: the ionic drawing of:func:`metal_disconnected_spelling_and_order`, the
    bonded drawing of an organometallic cation of:func:`metal_connected_spelling_and_order`, or
    the split drawing of a dative drawing and its ionic drawing), each named as an outermost
    ``name`` call of its own by ``fn(namer,...)``; the call returns the best result, with the
    provenance and records of the run that gave it (the run that gave it is run again last when it
    was not the last run).

    Better is the higher rank (:func:`form_result_rank`: the label, and for two failures the
    decline that says a name exists below the PIN, which a wider tier returns); on equal ranks a
    drawing that gives a name stands over one that gives none, and of two that give a name the one
    nearer the ionic drawing, so two drawings of one compound with one standard InChIKey get one
    name; when none gives a name, the drawing as written, as before. The result is never lower than
    the one the drawing as written gave, so no name that exists without the rewrite is lost or
    lowered. The name of another drawing is recorded as a non-PIN label
    (:func:`record_metal_rewrite_non_pin`) during its run. Each run goes through every gate, the
    round trip included, against the string it names; the standard InChIKey of that string is the
    input's, or, for a dative drawing, the reference key of its split drawing.

    A dative drawing has no standard InChIKey, so no name can be read back against it and the
    gates decline a name of the drawing as written (the key of the input is None). It is run last,
    and only when no other drawing names or declines with ``NO_VERIFIED_PIN``: the result of that
    run is the one the engine gave without the offer.

    Offer, not return (the contributor guide a project rule): a rewrite that fails, or lands below the
    drawing as written, ships nothing."""
    from .memo import pop_sandbox as _memo_pop_sandbox, push_sandbox as _memo_push_sandbox
    from ..metrics import provenance as _pv
    from ..namer import _LP_INPUT, _compose_atom_orders
    given = args[0] if args else kwargs.get('smiles')
    rest = tuple(args[1:])
    extra = {k: v for k, v in kwargs.items() if k != 'smiles'}
    scope_open = getattr(_LP_INPUT, 'open', False)
    entry_rewrite = getattr(_LP_INPUT, 'rewrite', None) if scope_open else None

    def run(string, string_order, rewritten):
        _pv.clear_provenance()
        if scope_open:
            # the entry that opened the input scope re-keys an atom map of the string
            # named to the caller's atoms: the string of this run is that string
            if string_order is None:
                _LP_INPUT.rewrite = entry_rewrite
            else:
                base = entry_rewrite[2] if entry_rewrite else None
                _LP_INPUT.rewrite = (
                    string, entry_rewrite[1] if entry_rewrite else given,
                    _compose_atom_orders(list(base) if base else None, list(string_order)),
                    bool(entry_rewrite and entry_rewrite[3]))
        # a memo scope of its own (see ``Orthonym._name_with_form_choice``): a value one run
        # memoised is not served to the next run of another string
        box = _memo_push_sandbox()
        rewrite_token = METAL_REWRITE_RUN.set(rewritten)
        split_token = METAL_DATIVE_SPLIT_RUN.set(rewritten and offer.keyless)
        try:
            out = fn(namer, string, *rest, **extra)
        except OrthonymLimitError as exc:
            out = exc
        finally:
            METAL_DATIVE_SPLIT_RUN.reset(split_token)
            METAL_REWRITE_RUN.reset(rewrite_token)
            _memo_pop_sandbox(box)
        return out

    def rank(string, out):
        if isinstance(out, OrthonymLimitError):
            return (0, 1 if out.code == "NO_VERIFIED_PIN" else 0)
        return form_result_rank(namer, string, out)

    def names(out):
        return isinstance(out, str) and bool(out) and not is_failure_name(out)

    def preference(result):
        # position 0 is the drawing as written, a larger position is nearer the ionic drawing
        position, _string, _order, out, result_rank = result
        is_name = names(out)
        return (result_rank, is_name, position if is_name else -position)

    entries = [(0, given, None)] + [(k, string, order) for k, (string, order) in
                                    enumerate(offer.drawings, 1)]
    if offer.keyless:
        entries = entries[1:] + entries[:1]
    owned = METAL_OFFER_OWNED.set(True)
    try:
        results = []
        for position, string, order in entries:
            if position == 0 and offer.keyless and any(r[4] > (0, 0) for r in results):
                continue
            out = run(string, order, position > 0)
            results.append((position, string, order, out, rank(string, out)))
        best = max(results, key=preference)
        result = best[3]
        last = results[-1]
        if best is not last and (best[3], best[4]) != (last[3], last[4]):
            # the run that gave the result is run again unless the last run gave the same
            # result: its provenance and records are then the ones the caller reads
            result = run(best[1], best[2], best[0] > 0)
        if isinstance(result, OrthonymLimitError):
            raise result
        return result
    finally:
        METAL_OFFER_OWNED.reset(owned)


def make_metal_offer_scope(owned_elsewhere):
    """The decorator of ``Orthonym.name`` that offers the other drawing of a metal drawing beside
    the drawing as written (:func:`name_with_metal_offer`) and keeps the better name.

    ``owned_elsewhere`` are the context variables of the other scopes that name strings of their
    own (the charge-form run and the strict PIN twin of ``namer``): while one is set, a call names
    the string it is given. The decorator is outside ``_form_choice_scope`` (each drawing gets its
    own charge-form choice), for the caller's own ``name`` call only: a nested call, a call
    inside a form's run or inside the offer itself call ``fn`` directly. An input without a metal
    drawing to rewrite is named as it always was."""
    def _scope(fn):
        @functools.wraps(fn)
        def _wrapper(self, *args, **kwargs):
            from .fragment_naming import name_scope_depth
            if (METAL_OFFER_OWNED.get() or any(flag.get() for flag in owned_elsewhere)
                    or name_scope_depth() > 0):
                return fn(self, *args, **kwargs)
            given = args[0] if args else kwargs.get('smiles')
            if not isinstance(given, str):
                if given is None or not _input_mol.handles(given):
                    return fn(self, *args, **kwargs)
                try:
                    given = _input_mol.smiles_for(given)
                except Exception:                  # fn raises it as it always did
                    return fn(self, *args, **kwargs)
                args, kwargs = ((given,) + tuple(args[1:]), kwargs) if args else (
                    args, {**kwargs, 'smiles': given})
            offer = metal_offer_for(given)
            if offer is None:
                return fn(self, *args, **kwargs)
            return name_with_metal_offer(self, fn, offer, args, kwargs)
        return _wrapper
    return _scope
