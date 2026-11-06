"""The skeleton of a labelled molecule named again with its hydro locants cited.

A fully saturated fused ring system is named by the count form of the hydro prefix with no locants
, the Blue Book, "Total hydrogenation is indicated by the appropriate multiplicative prefixes
indicating the total number of hydrogen atoms attached, but locants are omitted"): ``decahydronaphthalene``,
``hexadecahydro-1H-cyclopenta[a]phenanthrene``. OPSIN reads no descriptor on that form when two nuclides sit on one
atom (``decahydro(1,1-2H2)naphthalene`` is refused), so the decorator found no name for a labelled steroid or
decalin. (:44180): "if isotopic modification requires a locant to specify its position, then all
locants must be specified and none are omitted": the decorator runs once more, after every other try has declined,
with the hydro prefixes of the skeleton citing their locants
(``1,2,3,4,4a,5,6,7,8,8a-decahydro(5,5-2H2)naphthalene``, which OPSIN reads). The name is not the book's form of the
hydro prefix, so it is recorded below the PIN.
"""
import contextvars

from rdkit import Chem

from ..assembly.locant_omission import ISOTOPIC_HYDRO_LOCANTS, ISOTOPIC_STEREO_HYDRO_LOCANTS, ISOTOPIC_STEREO_LOCANTS

_ACTIVE: "contextvars.ContextVar[bool]" = contextvars.ContextVar("orthonym_isotope_hydro_locants", default=False)


def scope_reason() -> str:
    """The reason of the isotopic naming scope the decorator opens: the hydro-locants one inside the second try."""
    return ISOTOPIC_HYDRO_LOCANTS if _ACTIVE.get() else "isotope"


def stereo_scope_reason() -> str:
    """The reason of the stereo retry of the decorator (``assembly.locant_omission.scope_cites_stereo_locants``):
    inside the hydro-locants try the retry keeps the hydro locants that try cites."""
    return ISOTOPIC_STEREO_HYDRO_LOCANTS if _ACTIVE.get() else ISOTOPIC_STEREO_LOCANTS


def _has_saturated_fused_rings(mol) -> bool:
    """True when ``mol`` has a ring atom of two rings and an sp3 ring atom (a skeleton a hydro prefix can name)."""
    ri = mol.GetRingInfo()
    return (any(ri.NumAtomRings(a.GetIdx()) >= 2 for a in mol.GetAtoms())
            and any(a.IsInRing() and a.GetHybridization() == Chem.HybridizationType.SP3 for a in mol.GetAtoms()))


def _book_retry_to_come() -> bool:
    """True while a run that gave a book spelling is still to be run again with the mechanical spellings
    (``namer._name_with_book_retry``): a name that run finds is the one of the molecule before this try existed, so
    this try stands back for it and runs in the last rung, or in a run that gave no book spelling."""
    from ..assembly import book_prefixes as bp
    box = bp._FIRED.get()
    return bool(box and box[0] and bp.book_forms_enabled())


def decorate_with_hydro_locants(stripped, passes):
    """``passes`` (the decorator's passes on the skeleton) run with the hydro locants cited, or None."""
    from ..assembly.memo import fresh_scope, pop_sandbox
    from ..metrics.provenance import record_non_pin_label
    from ..opsin_capabilities import phane_names_withheld
    if not _has_saturated_fused_rings(stripped) or _book_retry_to_come():
        return None
    token = _ACTIVE.set(True)
    _fresh = fresh_scope()      # the caches of the first tries hold names made without the locants
    try:
        with phane_names_withheld():
            name = passes()
    finally:
        pop_sandbox(_fresh)
        _ACTIVE.reset(token)
    if name:
        record_non_pin_label(
            name, rule="P-31.2.3.3.2",
            detail="the hydro prefix of a fully saturated ring system cites its locants, which the book omits; "
                   "P-82.6.1.1 asks for the locants of the labelled positions")
    return name
