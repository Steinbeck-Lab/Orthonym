"""The probe name of the decomposition engine, read the way the PIN tier reads it for an ester molecule.

 "All preferred IUPAC names for esters are named by functional class nomenclature". The decomposition engine
splits a molecule only when the name the pipeline gives it without the split (``namer.name_pipeline_only``) is poor.
The ring cascade of a wider tier names an ester molecule with the ester as a prefix (acyloxy, alkoxycarbonyl:
'2-methyl-3-methylidene-2-{4-methyl-5-[(phenylacetyl)oxy]pent-3-en-1-yl}bicyclo[2.2.1]heptane'), a valid name that the
PIN tier declines, so at the PIN tier the probe gives no name and the ester is split ('2-methyl-5-(2-methyl-3-
methylidenebicyclo[2.2.1]heptan-2-yl)pent-2-en-1-yl phenylacetate').:func:`probe_name` is the probe of a wider tier read
the same way: such a name is no name for the split's decision, and it stays the molecule's name when the split names
nothing (the pipeline then names the molecule as before).

The cascade records the name (:func:`record_principal_ester_prefix`) while the probe watches; the record is kept in the
memo scope beside the probe's memoized name, so a probe served from the cache reads it too.
"""
import contextvars

#: the watch of:func:`probe_name`: a one-item list while a probe runs, else None
_WATCH: contextvars.ContextVar = contextvars.ContextVar("orthonym_ester_prefix_probe", default=None)
_NAMESPACE = "name_pipeline_only.ester_prefix"


def record_principal_ester_prefix() -> None:
    """The ring cascade named the molecule with its principal ester cited as a prefix (see the module docstring)."""
    seen = _WATCH.get()
    if seen is not None:
        seen[0] = True


def probe_watched() -> bool:
    """Whether a probe of:func:`probe_name` runs."""
    return _WATCH.get() is not None


def probe_name(smiles: str, style: str):
    """``namer.name_pipeline_only(smiles, style)``, or ``''`` at a wider tier for a name that cites the molecule's
    principal ester as a prefix."""
    from ..metrics.provenance import wider_tier_ctx
    from ..namer import name_pipeline_only
    from .memo import side_get, side_put
    from .nested_memo import renaming_state
    seen = [False]
    token = _WATCH.set(seen)
    try:
        name = name_pipeline_only(smiles, style=style)
    finally:
        _WATCH.reset(token)
    key = (smiles, style, renaming_state(), wider_tier_ctx.get())     # the key of the probe's own memo entry
    if seen[0]:
        side_put(_NAMESPACE, key, True)
    if name and wider_tier_ctx.get() and (seen[0] or side_get(_NAMESPACE, key)):
        return ''
    return name
