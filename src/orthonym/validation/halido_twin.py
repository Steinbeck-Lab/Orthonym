"""The twin of an additive organometallic name: its halide ligands spelled 'chloro', not 'chlorido'.

 Compounds with at least one metal-carbon single bond (the Blue Book) names a compound of a
Group 3-12 metal by its ligands cited in alphanumerical order, then the metal: '[Ti(CH3)Cl3]
trichlorido(methanido)titanium, trichlorido(methyl)titanium' (:39789). Definitions of terms (:39761):
"The names of anionic ligands, whether organic or inorganic, end in 'o'. If the anion name ends in 'ate',
'ite', or 'ide', the final letter 'e' is replaced by 'o', giving 'ato', 'ito', or 'ido'", and (:39767) "Because
of a long established tradition, names for substituent groups as defined in Section can also be used as
ligand names".

OPSIN 2.9.0 rejects the 'chlorido' ligand word ('trichlorido(methyl)titanium' has no parse) and reads
the long-established spelling of the same ligand, 'trichloro(methyl)titanium', to the sigma-bonded structure.
The name used to ship with no read-back at all (the ``organometallic_additive`` carve-out), and the sample of
the CO audit (2026-10-17) found no wrong name among 27; this twin is the check that makes that a measured
fact instead of a trust in the producer.

The twin is the name with each 'chlorido', 'bromido', 'fluorido', 'iodido' spelled 'chloro', 'bromo',
'fluoro', 'iodo'; 'hydrido' is a word OPSIN reads and stays. It is a rewrite of the tokens of the name, applied
to the additive names of the transition-metal producer only: a name of the shape of
``namer._ORGANOMETALLIC_ADDITIVE_PIN_RE`` without a hapto, kappa or mu descriptor (a name with one has no
spelling OPSIN reads, and ships as the deliberate eta-hapto carve-out). The standard InChIKey does not see the
bonds of a metal, so the read-back of the twin checks the ligands and their number; the producer certifies
the bonds (``rules.organometallics._sigma_additive_ligands_certified``).

The twin proves the structure and never the spelling: the producer owns the spelling.
"""
import re

_HALIDO = re.compile(r"(chlor|brom|fluor|iod)ido")
#: a hapto, kappa or mu descriptor: a name with one has no spelling OPSIN reads (the gate keeps it as the deliberate
#: carve-out ``organometallic_additive``, ``namer._final_opsin_validity_gate``); the one home of the pattern
HAPTO_DESCRIPTOR = re.compile(r"[ηκμ]")


def halido_twin(name: str) -> str:
    """The 'chloro' spelling of an additive organometallic name with 'chlorido' ligands, or ``name`` itself
    when it is no such name. Pure and deterministic."""
    if not name or HAPTO_DESCRIPTOR.search(name) or _HALIDO.search(name) is None:
        return name
    from ..namer import _ORGANOMETALLIC_ADDITIVE_PIN_RE
    if name not in _BONDED_HALIDO_NAMES and _ORGANOMETALLIC_ADDITIVE_PIN_RE.match(name) is None:
        return name
    return _HALIDO.sub(r"\1o", name)


#: the names with a halide ligand word that ``rules.organometallics._assemble_bonded_metal_name`` built for the metals
#: admitted to ``BONDED_NAMES`` (``data.metal_scope``): the producer owns the spelling and reports the string, so the
#: twin reads it for every such metal and no list of metal names is kept here (advisor ruling 2026-11-01: 'chlorido' is
#: the book's ligand word, 'chloro' only the read-back)
_BONDED_HALIDO_NAMES = set()


def note_bonded_halido_name(name: str) -> str:
    """Record ``name`` as an additive name of a bonded metal with halide ligand words; returns it unchanged."""
    if _HALIDO.search(name):
        _BONDED_HALIDO_NAMES.add(name)
    return name
