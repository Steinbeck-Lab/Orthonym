"""The probe name of the decomposition engine, read the way the PIN tier reads it for an ester molecule.

 "All preferred IUPAC names for esters are named by functional class nomenclature". The decomposition engine
splits a molecule only when the name the pipeline gives it without the split (``namer.name_pipeline_only``) is poor.
The ring cascade of a wider tier names an ester molecule with the ester as a prefix (acyloxy, alkoxycarbonyl:
'2-methyl-3-methylidene-2-{4-methyl-5-[(phenylacetyl)oxy]pent-3-en-1-yl}bicyclo[2.2.1]heptane'), a valid name that the
PIN tier declines, so at the PIN tier the probe gives no name and the ester is split ('2-methyl-5-(2-methyl-3-
methylidenebicyclo[2.2.1]heptan-2-yl)pent-2-en-1-yl phenylacetate').:func:`probe_name` is the probe of a wider tier read
the same way: such a name is no name for the split's decision, and it stays the molecule's name when the split names
nothing (the pipeline then names the molecule as before).

The name of a wider tier is no name, for an ester molecule, also when OPSIN reads it as another molecule
(:func:`names_another_molecule`): the decomposition engine would otherwise take the name for a good one and leave the
ester to the producers that cite it in a prefix. The engine asks only for a name its quality gate accepted, so that a
name it rejects anyway costs no OPSIN read.

The cascade records the name (:func:`record_principal_ester_prefix`) while the probe watches; the record is kept in the
memo scope beside the probe's memoized name, so a probe served from the cache reads it too.
"""
import contextvars

#: the watch of:func:`probe_name`: a two-item list while a probe runs, else None (the first item: the cascade cited the
#: principal ester as a prefix; the second: and the probe's name is no name at the PIN tier either, see
#::func:`record_principal_ester_prefix`)
_WATCH: contextvars.ContextVar = contextvars.ContextVar("orthonym_ester_prefix_probe", default=None)
_NAMESPACE = "name_pipeline_only.ester_prefix"
_NAMESPACE_PIN_TIER = "name_pipeline_only.ester_prefix_pin_tier"


def record_principal_ester_prefix(pin_tier: bool = False) -> None:
    """The ring cascade named the molecule with its principal ester cited as a prefix (see the module docstring). The
    probe's name is no name at a wider tier; with ``pin_tier`` it is none at the PIN tier either (the cascade's name of a
    polycyclic parent: the PIN tier has no other way to the functional class name of the ester)."""
    seen = _WATCH.get()
    if seen is not None:
        seen[0] = True
        seen[1] = seen[1] or pin_tier


def probe_watched() -> bool:
    """Whether a probe of:func:`probe_name` runs."""
    return _WATCH.get() is not None


def probe_name(smiles: str, style: str):
    """``namer.name_pipeline_only(smiles, style)``, or ``''`` at a wider tier for a name that cites the molecule's principal
    ester as a prefix (at the PIN tier too for the name of a polycyclic parent, ``record_principal_ester_prefix``)."""
    from ..metrics.provenance import wider_tier_ctx
    from ..namer import name_pipeline_only
    from .memo import side_get, side_put
    from .nested_memo import renaming_state
    seen = [False, False]
    token = _WATCH.set(seen)
    try:
        name = name_pipeline_only(smiles, style=style)
    finally:
        _WATCH.reset(token)
    key = (smiles, style, renaming_state(), wider_tier_ctx.get())     # the key of the probe's own memo entry
    if seen[0]:
        side_put(_NAMESPACE, key, True)
    if seen[1]:
        side_put(_NAMESPACE_PIN_TIER, key, True)
    if name and ((wider_tier_ctx.get() and (seen[0] or side_get(_NAMESPACE, key)))
                 or seen[1] or side_get(_NAMESPACE_PIN_TIER, key)):
        return ''
    return name


def names_another_molecule(name: str, mol) -> bool:
    """True at a wider tier when ``mol`` holds an ester and OPSIN reads ``name`` as a structure whose constitution is
    not that of ``mol``: the pipeline's name drops or invents atoms ('ethyl 4-(cyclohexyloxy)butanoate' for ethyl
    cyclohexyl butanedioate, an ether). The engine keeps the pipeline's name only when it is good enough, and a name of
    another molecule is no name of this one: the split is made, and the ester is named by functional class
    , not cited as a prefix of a name that no longer holds it. Scoped to esters, the class this module reads
     SENIORITY ORDER FOR CLASSES, Table 4.1: class 9 esters rank above class 11 amides, class 17 hydroxy compounds
    and class 19 amines, so the name that holds the ester only in a prefix is a name of a lower class); a carbamate is an
    ester of carbamic acid, named by its own route, the Blue Book), and is not counted.

    Three answers are possible of OPSIN (``decomposition.engine.name_constitution_verdict``); only "another molecule"
    rejects the name. A name OPSIN cannot judge (no JVM, the name not read, the coverage oracle off) stays, as in the
    engine's own contract: the oracle there only ever rescues a name, it is not a reason to lose one."""
    from ..metrics.provenance import wider_tier_ctx
    if not name or mol is None or not wider_tier_ctx.get():
        return False
    from ..decomposition.engine import name_constitution_verdict
    from ..namer import _validity_gate_jar_present
    from ..rules.esters import non_carbamate_ester_carbonyl_atoms
    if not _validity_gate_jar_present() or not non_carbamate_ester_carbonyl_atoms(mol):
        return False
    return name_constitution_verdict(name, mol) is False
