"""Which metals the engine names beside organic ions, and by which mechanism.

One table, ``METAL_SCOPE``. The key is an element symbol. The value says which naming
mechanisms the metal is admitted to (``mechanisms``) and which charges a cation of the metal
may carry (``valences``). A metal without an entry is admitted to no mechanism, so a salt or
an adduct that holds it is not composed from its ions. Admitting a metal to a mechanism is an
edit of this table, followed by a panel measurement; the producers behind a mechanism still hold
per-metal data of their own (element names, stems, the organyl rules of Group 12), and the panel
measurement is what shows that a new metal has an entry in each.

Mechanisms (each is a constant below; a task that builds a mechanism adds its constant):

``ION_SALTS``
    a net-neutral salt, or a charged assembly, of the metal cation with organic ions is named
    from its ions: a binary salt or an adduct at the best-effort tier.
``CARBANION_SALTS``
    the same for an assembly that also holds a carbanion or a hydride. Such an assembly is an
    organometallic compound for the metal; the Blue Book's definition is "Organometallic
    compounds are compounds having at least one bond between one metal atom and one carbon
    atom." INTRODUCTION). A metal without this mechanism is never composed with a
    carbanion or a hydride. The hydride ion is cited as 'hydride' in the salt of such a metal
    (``rules.metal_atoms.inorganic_anion_word``), the salt of an anion with a cation 'followed by
    that of the anion', the Blue Book).
``ELEMENT``
    the atom, the ions and the isotopes of the metal are named by the element name, the charge
    number of an ion and the nuclide symbol of an isotope ('zinc', 'zinc(2+)', '(65Zn)zinc'): the
    charge number is the notation of 'pentaammine(ethyl)osmium(1+) chloride',
    the Blue Book) and the nuclide symbol is "enclosed in parentheses... before the part of
    the compound that is isotopically substituted",:43718). The Blue Book gives such a
    name no PIN: PREFERRED IUPAC NAMES (:2056) labels "the names of compounds whose parent
    hydride contains at least one of the following elements: B, Si,... At, and that also contain at
    least one carbon atom in their structure". The producer is ``rules.metal_atoms``; the class of
    the dispatch table is ``METAL_ATOM``.
``DISCONNECT``
    a drawing in which the metal is joined to non-carbon atoms by single bonds is named through
    its ionic drawing, the one a charge-conserving heterolytic disconnection of those bonds gives
    (the metal gains one positive charge per bond, the other atom one negative charge), when both
    drawings have the same standard InChIKey. Structures for Lewis 'adducts', "One of
    the fundamental principles of organic chemical nomenclature is to name the structure that is
    drawn", so the drawing as written is named as well and the better of the two names ships.
``CONNECT_ANION``
    a drawing in which a cation of the metal carries a metal-carbon bond and is drawn beside the
    free anion that neutralises it (``[Cl-].[Zn+]c1ccccc1``) is named through its bonded drawing,
    the one that joins the donor atom of the anion to the metal (``Cl[Zn]c1ccccc1``), when both
    drawings have the same standard InChIKey: the reverse of ``DISCONNECT``, for the bond to the
    anion and not for the bond to carbon. The Blue Book names a compound with a bond between a
    metal and carbon as an organometallic compound INTRODUCTION), and asks for
    the structure that is drawn, so the drawing as written is named as well and the better of the
    two names ships.

``BONDED_NAMES``
    a drawing in which the metal is bonded to terminal atoms, to organyl groups and to hydrogen, or
    carries a charge, is named as a complex: "Compounds are named by citing the names of ligands,
    including hydrogen atom(s), in alphanumerical order, followed by the name of the metal. The
    presence of hydrogen attached to a metal atom must always be indicated by the prefix 'hydrido'"
     Compounds with at least one metal-carbon single bond), and "names for substituent
    groups as defined in Section can also be used as ligand names" Definitions of
    terms). The terminal ligands are the prefixes of ``TERMINAL_LIGAND_PREFIXES`` in
    ``data/organometallics.py`` (hydrido, hydroxy, sulfanyl, amino, oxo, sulfanylidene); a charge is
    the closing charge number of the complex, as in 'pentaammine(ethyl)osmium(1+)'. The
    producer is ``rules.organometallics._assemble_bonded_metal_name``. A halide beside a multiply bonded ligand is
    cited with the book's ligand word ('dichlorido(methylidene)titanium',: an anion name ending in 'ide' ends
    in 'ido' as a ligand; the active OPSIN reads it through its 'chloro' twin, ``validation.halido_twin``), and a
    multiplied halide, hydrido, oxo, hydroxy or amino ligand takes no enclosing marks ('benzylidenedichloridoruthenium',
    . The Blue Book gives no PIN for
    such a compound: "Coordination nomenclature is the primary nomenclature method used to name
    organometallic compounds containing elements of Groups 3 through 12", and
    PREFERRED IUPAC NAMES gives the label "PIN" to "the names of compounds whose parent hydride
    contains at least one of the following elements: B, Si, Ge, Sn, Pb, N, P, As, Sb, Bi, O, S, Se,
    Te, Po, F, Cl, Br, I, At, and that also contain at least one carbon atom in their structure":
    zinc is none of those elements.

``AMMINE``
    a neutral molecule of ammonia in an assembly whose every true metal is admitted here is cited as
    the component 'ammonia' of a adduct ('ammonia—zinc(2+) (4/1)' for ``[Zn+2].N.N.N.N``).
    The table of the other single-atom components holds no bare nitrogen, because a bare nitrogen
    fragment is more often a perception artefact than an ammonia of crystallisation; beside a metal
    that is admitted here it is the ligand of an ammine complex. The Blue Book gives such a name no
    PIN: "preferred IUPAC names cannot be assigned to mixed adducts because preferred IUPAC names
    have not yet been determined for inorganic components" Mixed organic - inorganic
    adducts), and "The common names water, ammonia,... are used in these recommendations but their
    use as preferred IUPAC names is deferred... thus, no PIN label will be assigned in names
    including them (see " Retained names). The coordination name cites the ligand
    as 'ammine' Definitions of terms); OPSIN 2.9.0 does not read it, so the adduct cites the
    name of the molecule. The producer is ``rules.metal_atoms.ammine_component_names``, read by
    ``rules.adducts.name_adduct`` in a call with ``general_fallback_unverified`` (the best-effort tier).

``MACROCYCLE``
    the porphyrin dianion of a complex of the metal is cited on the porphyrin parent, as
    ``porphyrin-21,23-diide`` with the substituent prefixes before it ('zinc
    5,10,15,20-tetraphenylporphyrin-21,23-diide'), in an assembly whose every true metal is admitted
    here. "An anion derived formally by the removal of one or more hydrons from any position of a
    neutral parent hydride is preferably named by using the suffix '-ide'... Numerical prefixes 'di',
    'tri', etc. are used to denote multiplicity; locants identify positions of the negative charges"
     Anions derived from parent hydrides and their derivatives, the Blue Book), and the parent hydride
    ``porphyrin`` implies its two N-H on N21 and N23 (the tetrapyrrole recommendations.1 that
     cites). The Blue Book gives such a name no PIN: "Preferred IUPAC names (PINs) are not
    identified for the compounds in this Chapter" INTRODUCTION,:50943; the porphyrin parent
    belongs to, and this sentence covers magnesium as well), and "Coordination nomenclature is the primary
    nomenclature method used to name organometallic compounds containing elements of Groups 3 through 12",
    :39753). The producer is
    ``rules.porphyrin.name_porphyrin_dianion``; ``rules.metal_macrocycles`` reads this table, and
    ``rules.salts.name_salt`` and ``rules.adducts.name_adduct`` read that module, at the tiers that
    name ions (the best-effort flags). Without the mechanism the ligand is named by the general
    engine.

``DATIVE``
    a drawing in which the metal is the acceptor of dative bonds (``CC(=O)[O-]->[Zn+2]<-[O-]C(C)=O``)
    is named through the same drawing without those bonds, the split drawing: the donor keeps its
    lone pair, the metal keeps its charge, and nothing else changes. RDKit gives a drawing with
    dative bonds no standard InChIKey, so the split drawing's key is the reference key of the
    name's read-back, and an independent count of atoms and charges between the two drawings
    (``perception.metals.atom_and_charge_ledger_ok``, no charge moved) ties it to the drawing
    given. Structures for Lewis 'adducts' draws the 'dative' bond as structure (I) and
    says "One of the fundamental principles of organic chemical nomenclature is to name the
    structure that is drawn", so a name reached through the split drawing is not the name of the
    structure drawn: it is labelled below the PIN, and the default tier declines it. The split
    drawing is then treated as any drawing (``DISCONNECT`` applies to its metal-heteroatom single
    bonds). The row says how the name was checked: ``verified`` is ``opsin_split``.

``ZERO_VALENT``
    an assembly in which a neutral atom of the metal stands beside charged components (``[Ir]`` beside
    three carbanions, ``[Zn].[Cl-].[Cl-]``) is named as a adduct of its components with the atom
    cited with the oxidation number zero ('iridium(0)'), at the best-effort tier only: a reader of
    'zinc—chloride (1/2)' balances the charges itself and reads zinc(2+), 'zinc(0)—chloride (1/2)'
    reads the neutral atom drawn. User decision 2026-10-30: such literal names are wanted for PubChem's
    neutral-metal drawings (it replaces decision 2 of 2026-10-13, no producer for a neutral atom beside
    anions). The producer is ``rules.adducts._zero_metal_name``, tried only when the name without the zero
    does not read back to the drawing; the name is never a PIN, the Blue Book). The metals
    admitted are the ones the panel of the lane measured (`internal notes`).

``PARENT_HYDRIDE_IONS``
    a cation of the metal whose bonds and hydrogens make up one less than its standard bonding number is named on
    the mononuclear parent hydride of the metal, as a silicon or germanium centre is: "silane, SiH4, germane, GeH4,
    stannane, SnH4, and plumbane, PbH4... by replacing the 'ane' ending... by the suffix 'ylium'"
    Cationic centers in parent hydrides, the Blue Book; 'triphenylsilylium (PIN)',:41557). The Blue Book
    names the organic compounds of tin and lead substitutively, on 'stannane' and 'plumbane', and gives them
    PINs, so such a metal is admitted to the mechanisms that keep the substitutive name: this one,
    ``CONNECT_ANION`` (the bonded drawing of an organotin cation beside its anions is a substituted stannane) and
    ``ZERO_VALENT``; never to ``DISCONNECT`` or ``BONDED_NAMES``, whose coordination names the book does not use for
    these elements. The producer is ``rules.group14_centres.emit_group_ylium``.

A metal is ``complete`` when the plan of its phase is done: every drawing of its compounds (ionic,
bonded, dative, an atom or an ion) is attempted by the mechanisms above, so an input of it that no
producer names fails for want of a name and not for want of support of the element. The failure
label of such an input is ``UNNAMEABLE`` and not ``UNSUPPORTED_ELEMENT``
(``errors.classify_failure_limit``, through ``has_complete_scope``). The elements of Groups 1 and 2
are admitted to ``ION_SALTS``, ``CARBANION_SALTS`` and ``ZERO_VALENT`` (magnesium also to ``MACROCYCLE`` and
``DISCONNECT``) and are not complete: their failures keep their label.

The valences are the charges of the cation that a rewrite of the structure may give the metal
(a heterolytic disconnection of a metal-ligand bond adds one positive charge to the metal; the
cation that a metal with its bonds would give if every one of them were broken is its formal
charge plus its number of bonds). The Group 1 elements carry +1 and the Group 2 elements +2; zinc
carries +2. The metals of ``BONDED_METAL_VALENCES`` (Al, Hg, Pd, Pt, Ru, Ti, V, W) and of
``MORE_BONDED_METAL_VALENCES`` (38 more) are admitted to every mechanism zinc has, with the charges listed there,
and are not complete; those of ``RETAINED_ION_METALS`` not to ``ELEMENT``.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable

ION_SALTS = "ion_salts"
CARBANION_SALTS = "carbanion_salts"
DISCONNECT = "disconnect"
CONNECT_ANION = "connect_anion"
BONDED_NAMES = "bonded_names"
ELEMENT = "element"
AMMINE = "ammine"
MACROCYCLE = "macrocycle"
DATIVE = "dative"
ZERO_VALENT = "zero_valent"
PARENT_HYDRIDE_IONS = "parent_hydride_ions"


@dataclass(frozen=True)
class MetalScope:
    """The mechanisms a metal is admitted to and the cation charges it may carry."""
    mechanisms: FrozenSet[str]
    valences: FrozenSet[int]
    complete: bool = False


#: the elements of Groups 1 and 2: the Blue Book's INTRODUCTION counts them with the
#: transition elements for the status of the names of organometallic compounds
GROUP_1 = ("Li", "Na", "K", "Rb", "Cs", "Fr")
GROUP_2 = ("Be", "Mg", "Ca", "Sr", "Ba", "Ra")

#: the metals admitted to ``ZERO_VALENT`` alone: the 28 of the 30 most frequent metals of PubChem's
#: neutral-atom drawings that the panel of the lane named with no wrong name (Ar and Bi named none)
ZERO_VALENT_METALS = ("Ir", "Y", "Pt", "Fe", "W", "Zr", "Ti", "V", "Pd", "Hf", "Co", "U", "Ni", "Al", "Cu",
                      "Ru", "Re", "Rh", "Sn", "Mn", "Mo", "Cr", "Rf", "Au", "Pb", "Ac", "Ag", "Fm")

#: every mechanism above: the ones zinc is admitted to
ALL_MECHANISMS = frozenset({DISCONNECT, CONNECT_ANION, BONDED_NAMES, ELEMENT, AMMINE, MACROCYCLE, DATIVE, ION_SALTS,
                            CARBANION_SALTS, ZERO_VALENT})

#: the metals of PubChem's bonded drawings admitted to every mechanism, with the charges of their cations: the
#: lane of 2026-10-30 measured them on a panel of their bonded, dative and ionic drawings (`internal notes
#: metal-bonded/``) with no wrong name. They are not ``complete``: their eta and multiply bonded ligands have no
#: producer yet, so their failures keep their label.
BONDED_METAL_VALENCES = {"Al": (3,), "Hg": (1, 2), "Pd": (2,), "Pt": (2, 4), "Ru": (2, 3), "Ti": (3, 4),
                         "V": (3, 4, 5), "W": (4, 6)}

#: the metals of the second panel of the lane (2026-10-30, 38 metals of PubChem's unnamed metal drawings, no wrong
#: name), and uranium (2026-10-31): every mechanism zinc has, with the charges of their cations. Not complete.
MORE_BONDED_METAL_VALENCES = {"Zr": (4,), "Hf": (4,), "Fe": (2, 3), "Co": (2, 3), "Ni": (2,), "Cu": (1, 2),
                              "Mo": (2, 3, 4, 5, 6), "Cr": (2, 3, 6), "Rh": (1, 3), "Re": (1, 3, 5, 7),
                              "Mn": (2, 3), "Ir": (1, 3), "Y": (3,), "Os": (2, 3, 4), "Ag": (1,),
                              "Au": (1, 3), "Cd": (2,), "Ga": (3,), "In": (3,), "Tl": (1, 3), "Nb": (5,),
                              "Ta": (5,), "Sc": (3,), "La": (3,), "Ce": (3, 4), "Sm": (3,), "Eu": (2, 3),
                              "Gd": (3,), "Nd": (3,), "Yb": (2, 3), "Lu": (3,), "Er": (3,), "Dy": (3,), "Tb": (3,),
                              "Pr": (3,), "Ho": (3,), "Tm": (3,), "U": (3, 4, 5, 6)}

#: metals whose ions have a retained name of their own ('copper(II)', 'iron(III)': ``data.ion_retained_names``,
#: INORGANIC_CATIONS) are not admitted to ``ELEMENT``: its 'copper(2+)' would replace the retained name and its label
RETAINED_ION_METALS = frozenset({"Ag", "Au", "Co", "Cr", "Cu", "Fe", "Mn", "Ni"})

METAL_SCOPE: Dict[str, MetalScope] = {
    # the hydride of a Group 1 or 2 salt (``[Na+].[H-]``), its carbanions and the neutral atom beside ions
    # ('potassium(0)'): the lane of 2026-10-30
    **{symbol: MetalScope(frozenset({ION_SALTS, CARBANION_SALTS, ZERO_VALENT}), frozenset({1})) for symbol in GROUP_1},
    **{symbol: MetalScope(frozenset({ION_SALTS, CARBANION_SALTS, ZERO_VALENT}), frozenset({2})) for symbol in GROUP_2},
    # magnesium porphyrins (the chlorophyll core) are named on the porphyrin dianion like zinc's
    "Mg": MetalScope(frozenset({ION_SALTS, CARBANION_SALTS, ZERO_VALENT, MACROCYCLE, DISCONNECT}), frozenset({2})),
    **{symbol: MetalScope(frozenset({ZERO_VALENT}), frozenset()) for symbol in ZERO_VALENT_METALS},
    **{symbol: MetalScope(ALL_MECHANISMS, frozenset(valences)) for symbol, valences in BONDED_METAL_VALENCES.items()},
    **{symbol: MetalScope(ALL_MECHANISMS - ({ELEMENT} if symbol in RETAINED_ION_METALS else frozenset()),
                          frozenset(valences))
       for symbol, valences in MORE_BONDED_METAL_VALENCES.items()},
    "Zn": MetalScope(ALL_MECHANISMS, frozenset({2}), complete=True),
    # Group 14 metals named on their parent hydrides: the lane of 2026-10-30
    **{symbol: MetalScope(frozenset({PARENT_HYDRIDE_IONS, CONNECT_ANION, ZERO_VALENT}), frozenset({4}))
       for symbol in ("Sn", "Pb")},
}


def has_mechanism(symbol: str, mechanism: str) -> bool:
    """True when the metal ``symbol`` is admitted to ``mechanism``; False for an unknown symbol."""
    scope = METAL_SCOPE.get(symbol)
    return scope is not None and mechanism in scope.mechanisms


def has_valence(symbol: str, charge: int) -> bool:
    """True when a cation of the metal ``symbol`` may carry ``charge``; False for an unknown symbol."""
    scope = METAL_SCOPE.get(symbol)
    return scope is not None and charge in scope.valences


def is_named_metal(symbol: str) -> bool:
    """True when the metal ``symbol`` is admitted to at least one mechanism other than ``ZERO_VALENT`` (a
    metal admitted to that one alone has no name for its ions, atoms or complexes: its failures keep the
    label ``UNSUPPORTED_ELEMENT``)."""
    scope = METAL_SCOPE.get(symbol)
    return scope is not None and bool(scope.mechanisms - {ZERO_VALENT})


def is_complete_metal(symbol: str) -> bool:
    """True when the metal ``symbol`` is ``complete`` (see the module docstring); False for an unknown symbol."""
    scope = METAL_SCOPE.get(symbol)
    return scope is not None and scope.complete


def has_complete_scope(symbols: Iterable[str]) -> bool:
    """True when every element of ``symbols`` is a metal admitted to at least one mechanism and at least
    one of them is complete: an input made of these elements that no producer names is the engine's gap,
    not an unsupported element. False for an empty set, for an element the table does not know and for
    metals none of which is complete (the elements of Groups 1 and 2 alone)."""
    symbols = set(symbols)
    return (bool(symbols) and all(is_named_metal(s) for s in symbols)
            and any(is_complete_metal(s) for s in symbols))
