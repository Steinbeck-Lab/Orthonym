"""Which OPSIN reading the read-back of a name may use: with ``-r`` or without it.

OPSIN's ``-r`` (``allowRadicals``) lets a name with unfilled valences ('(phosphoryl)oxy' for
``-O-P=O``: ``phosphoryl`` is ``-PO<``, three free valences) parse into a structure. OPSIN's SMILES of
such a name does not mark the radical: an atom that has a lower valence of its own (P(III) here) is
read as a closed-shell atom, so the name's structure compares equal to the closed-shell input by
InChIKey (measured: 1,436 of 1,438 closed-shell PubChem rows and 60 default-tier PINs shipped such a
name, with the ``-r`` read-back as the only reading that accepted it). A read-back that always passes
``-r`` therefore accepts a name that
describes a radical (or a substituent group) as the name of a closed-shell molecule.
"GENERAL METHODOLOGY" (the Blue Book): "Radicals are named by modifying a parent hydride name
to signal the subtraction or addition of one or more hydrogen atoms"; "The suffixes 'yl' (-H.),
'ylidene' (-2H.), 'ylidyne' (-3H.) denote the removal of hydrogen atoms". A name with a free valence
is such a name; the standard InChIKey sees neither the radical nor the bond order (see
``validation/radical_identity.py``), so only the reading of the name can refuse it.

The rule applied here: a name is read back WITH ``-r`` only when the input structure has radical
electrons (RDKit ``Atom.GetNumRadicalElectrons > 0`` on any atom), the only inputs whose names may
cite a free valence. The input is the SMILES given to the INNERMOST live ``name`` frame of the
thread (``fragment_naming.innermost_name_input``): the engine names the radical fragments of a
closed-shell input through nested ``name`` calls (the nitro group is named as ``O=[N+][O-]``), and
those frames read for their own fragment. After the outermost frame has returned, the labels that
``name_tiered`` sets read for its input (``reading_base_for``). Outside any ``name`` call and
``name_tiered`` there is none and the legacy reading (``-r``) stays, so a direct use of the oracle,
a script or a test of it is unchanged.

Two checks know their molecule and say so. ``OpsinOracle.rt_safe`` reads substituent-subtree strings,
which carry free valences by design (``legacy_reading``). ``charged_router._full_inchikey_rt_ok(mol,
name)`` verifies the radical fragments that ``route_charged`` names for a closed-shell input
(``dimethylarsanyl`` for ``C[As](C)c1ccnc2ccccc12``): its reading follows ``mol`` (``reading_for``);
without that, the default tier lost the PIN '4-(dimethylarsanyl)quinoline' (measured on the 3,243
Blue Book rows).
"""
import contextlib
import contextvars
from functools import lru_cache

# Set by ``reading_for``: the reading of one check whose molecule is known, in force over the
# input of the thread's outermost ``name``.
_FORCED = contextvars.ContextVar("orthonym_read_back_radicals", default=None)
# Set by ``reading_base_for``: the input of a ``name_tiered`` call, read when no ``name`` frame is
# live (the label logic that runs after ``name`` returned).
_BASE = contextvars.ContextVar("orthonym_read_back_base_input", default=None)


@lru_cache(maxsize=8192)
def smiles_has_radical_electrons(smiles: str) -> bool:
    """True when ``smiles`` has an atom with radical electrons (or cannot be read, the legacy
    reading: no verdict is a verdict for ``-r``)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True
    return any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms())


def input_allows_radicals() -> bool:
    """True when the read-back of a name may use OPSIN's ``-r`` for the input being named."""
    forced = _FORCED.get()
    if forced is not None:
        return forced
    from ..assembly.fragment_naming import innermost_name_input
    smiles = innermost_name_input() or _BASE.get()
    return True if not smiles else smiles_has_radical_electrons(smiles)


def cache_key(name: str):
    """The key of ``name`` in an oracle's parse caches: the name itself under the legacy reading,
    ``(name, False)`` under the closed-shell reading, so the two readings never share an entry."""
    return name if input_allows_radicals() else (name, False)


@contextlib.contextmanager
def reading_for(mol):
    """The read-back inside the block is made for ``mol`` (a check that knows its molecule, e.g.
    the radical fragment ``route_charged`` names for a closed-shell input), not for the input of
    the outermost ``name``: ``-r`` iff ``mol`` has radical electrons."""
    token = _FORCED.set(any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()))
    try:
        yield
    finally:
        _FORCED.reset(token)


@contextlib.contextmanager
def legacy_reading():
    """The read-back inside the block is made for a name that carries free valences by design (the
    substituent-subtree strings of ``OpsinOracle.rt_safe``): ``-r``, whatever the input."""
    token = _FORCED.set(True)
    try:
        yield
    finally:
        _FORCED.reset(token)


@contextlib.contextmanager
def reading_base_for(smiles):
    """``name_tiered`` of ``smiles``: the label logic that runs after ``name`` returned (the full
    round trip that promotes a name, the stereo comparison) reads for this input. A live ``name``
    frame still reads for its own input."""
    token = _BASE.set(smiles)
    try:
        yield
    finally:
        _BASE.reset(token)
