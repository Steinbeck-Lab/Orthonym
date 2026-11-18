"""What the OPSIN in use can read.

OPSIN 2.9.0 reads no phane names, no halogen "-uide" anions with substituents
(``diphenyliodanuide``) and no ring assembly names with composite locants
(``1^1,2^1:2^4,3^1-terphenyl``); OPSIN's main branch reads all three. A name of such a form can be
checked by a round trip only when the OPSIN in use reads it, so a producer of such names asks here
before it offers one (``phane_enabled``, ``composite_assembly_enabled``).

The release jar (the pinned, checksummed 2.9.0) is known to read neither and is not probed. Any
other jar -- built from OPSIN's main branch (``ORTHONYM_OPSIN``) or given by ``ORTHONYM_OPSIN_JAR`` --
is probed once with two names through the in-process OPSIN, the entry point of the validity
gate, and the answer is kept per jar (and, for a built jar, in its ``.json`` file). A probe that
cannot be answered (no in-process OPSIN, a busy JVM) reads as "cannot" for that call only and is
not kept, so one failed call never switches a capability off for the rest of the process.
"""
import contextlib
import contextvars
import json
from pathlib import Path
from typing import Dict

from . import jars

#: OPSIN 2.9.0 has a numerical term up to 50 ('pentaconta') and none above: a name with a larger count word has no
#: reading (``rules.hydrocarbon_series``, ``rules.replacement_chain_prefix``, ``validation.long_count_read_back``)
MAX_READABLE_COUNT = 50

#: capability -> (a name only newer OPSIN reads, its structure as SMILES, read with OPSIN's radical setting)
#: "radical_adduct": an adduct name with a radical component "Organic adducts", the Blue Book), which
#: OPSIN 2.9.0 reads as a substituent and the main branch reads as the mixture
_PROBES = {
    "phane": ("1,4(1,4)-dibenzenacyclohexaphane", "C1Cc2ccc(cc2)CCc2ccc1cc2", False),
    "halogen_uide": ("diphenyliodanuide", "[I-](c1ccccc1)c1ccccc1", False),
    "composite_assembly": ("1^1,2^1:2^4,3^1-terphenyl", "c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", False),
    "radical_adduct": ("ethane\u2014boranylidyne (1/1)", "[B].CC", True),
    #: "axial": an aryl-aryl axis descriptor without a locant,:49797; dan2097/opsin#353), which OPSIN
    #: 2.9.0 does not read and the main branch reads (the key cannot see the axis: ``validation.axis_identity``)
    "axial": ("(M)-[1,1'-binaphthalene]-2,2'-diol", "Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12", False),
    #: "aminiumyl": a cationic prefix built on an amine cation:42296,:42473
    #: '(N,N-dimethylmethanaminiumyl)acetate (PIN)'; dan2097/opsin#355), which OPSIN 2.9.0 does not read
    "aminiumyl": ("(N,N-dimethylmethanaminiumyl)acetate", "C[N+](C)(C)CC([O-])=O", False),
    #: "polyester": the functional-class name of a polyol ester of different acids method (1),:31840
    #: 'propane-1,2,3-triyl 1,2-diacetate 3-propanoate (PIN)'; dan2097/opsin#354), which OPSIN 2.9.0 does not read
    "polyester": ("propane-1,2,3-triyl 1,2-diacetate 3-propanoate", "CCC(=O)OCC(OC(C)=O)COC(C)=O", False),
}
_NONE: Dict[str, bool] = {key: False for key in _PROBES}
_CACHE: Dict[str, Dict[str, bool]] = {}


def _reset_cache() -> None:
    """Forget the answers kept for this process (tests)."""
    _CACHE.clear()


def _same_structure(got: str, want: str) -> bool:
    from rdkit import Chem
    g, w = Chem.MolFromSmiles(got) if got else None, Chem.MolFromSmiles(want)
    return g is not None and w is not None and Chem.MolToInchiKey(g) == Chem.MolToInchiKey(w)


def _probe(path: str) -> Dict[str, bool]:
    """Read the probe names with the in-process OPSIN (the jar at ``path``); raise
    ``jvm_bridge.OpsinUnavailable`` when it cannot answer."""
    from .jvm_bridge import OpsinUnavailable, opsin_stdout
    answer = {}
    for key, (name, smiles, radical) in _PROBES.items():
        text, served = opsin_stdout(name, radical, path)
        if not served:
            raise OpsinUnavailable(f"in-process OPSIN did not read the probe name {name!r}")
        answer[key] = _same_structure((text or "").strip(), smiles)
    return answer


def _sidecar(path: str) -> Path:
    return Path(path).with_suffix(".json")


def _stored(path: str):
    from .opsin_build import trusted_file
    sidecar = _sidecar(path)
    if not trusted_file(sidecar):
        return None
    try:
        data = json.loads(sidecar.read_text()).get("capabilities")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(data, dict) and set(data) == set(_PROBES) and all(isinstance(v, bool) for v in data.values()):
        return dict(data)
    return None


def _store(path: str, answer: Dict[str, bool]) -> None:
    """Keep ``answer`` in the ``.json`` file of a built jar; best effort, never an error. Only a sidecar that
    is a trusted file is read or rewritten, and it is rewritten through a new file of a unique name."""
    from .opsin_build import _write_json, trusted_file
    sidecar = _sidecar(path)
    if not trusted_file(sidecar):
        return
    try:
        data = json.loads(sidecar.read_text())
        if isinstance(data, dict):
            data["capabilities"] = answer
            _write_json(sidecar, data)
    except (OSError, ValueError):
        pass


def opsin_capabilities() -> Dict[str, bool]:
    """``{"phane": bool, "halogen_uide": bool, "composite_assembly": bool, "radical_adduct": bool, "axial": bool, "aminiumyl": bool, "polyester": bool}`` for the OPSIN
    this process uses."""
    from .jvm_bridge import OpsinUnavailable
    try:
        active = jars.active_opsin()
    except jars.JarUnavailable:
        return dict(_NONE)
    if active["label"] == "2.9.0" or not active["path"]:
        return dict(_NONE)
    path = active["path"]
    if path not in _CACHE:
        answer = _stored(path) if active["label"] == "main" else None
        if answer is None:
            try:
                answer = _probe(path)
            except OpsinUnavailable:
                return dict(_NONE)
            if active["label"] == "main":
                _store(path, answer)
        _CACHE[path] = answer
    return dict(_CACHE[path])


_PHANE_WITHHELD = contextvars.ContextVar("orthonym_phane_names_withheld", default=False)


@contextlib.contextmanager
def phane_names_withheld():
    """Scope of naming a skeleton that a later step writes into: inside it ``phane_enabled`` is False, so no phane
    producer offers a name. The isotope decorator, ``rules.isotopes``) inserts descriptors into the name of
    the isotope-stripped skeleton; the phane producers cite no isotopic modification, so a phane skeleton name cannot
    be labelled and the molecule would be left without a name, where the substitutive name of the same skeleton
    takes the descriptors. The functional class names of the sulfur, selenium and tellurium oxoacid halides
    ('sulfurous dichloride', ``rules.chalcogen_acid_halides``) are withheld in the same scope
    (:func:`label_free_names_withheld`): OPSIN reads no descriptor on their oxygen or class words."""
    token = _PHANE_WITHHELD.set(True)
    try:
        yield
    finally:
        _PHANE_WITHHELD.reset(token)


def label_free_names_withheld() -> bool:
    """True inside:func:`phane_names_withheld`: the producers whose names take no isotopic descriptor offer none."""
    return _PHANE_WITHHELD.get()


def phane_enabled() -> bool:
    """True when the OPSIN in use reads phane names (and no scope withholds the phane producers)."""
    return not _PHANE_WITHHELD.get() and opsin_capabilities()["phane"]


def composite_assembly_enabled() -> bool:
    """True when the OPSIN in use reads ring assembly names with composite locants. A capability set
    without the entry (a test that stands in for OPSIN 2.9.0) reads as "cannot"."""
    return bool(opsin_capabilities().get("composite_assembly", False))


def aminiumyl_enabled() -> bool:
    """True when the OPSIN in use reads an '-aminiumyl' prefix ('(N,N-dimethylmethanaminiumyl)acetate'). A capability
    set without the entry (an older sidecar, a test that stands in for OPSIN 2.9.0) reads as "cannot"."""
    return bool(opsin_capabilities().get("aminiumyl", False))


def polyester_enabled() -> bool:
    """True when the OPSIN in use reads the functional-class name of a polyol ester of different acids
    ('propane-1,2,3-triyl 1,2-diacetate 3-propanoate'). A capability set without the entry (an older sidecar, a test
    that stands in for OPSIN 2.9.0) reads as "cannot"."""
    return bool(opsin_capabilities().get("polyester", False))


def radical_adduct_enabled() -> bool:
    """True when the OPSIN in use reads an adduct name with a radical component ('ethane\u2014boranylidyne (1/1)').
    A capability set without the entry (an older sidecar, a test that stands in for OPSIN 2.9.0) reads as "cannot"."""
    return bool(opsin_capabilities().get("radical_adduct", False))
