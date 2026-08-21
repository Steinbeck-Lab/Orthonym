"""v25 G1: E1 atom-coverage certificate.

Verifies a GeneralEngineResult's atom->token partition: every heavy atom of
the molecule is bound by EXACTLY ONE token, every token occurs in the
emitted name, and (G1 scope) the molecule is neutral. Pure Python +
RDKit -- no Java, no OPSIN. This is the load-bearing "no silent atom drop"
gate the legacy count-based validation/atom_coverage.py never was; that
legacy module is untouched and unrelated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class E1Verdict:
    ok: bool
    reason: str


# --- F-E1: per-token element soundness (sound-by-refusal) -------------------
# E1 verifies the atom PARTITION but not that a token's CHEMISTRY matches the
# atoms it claims: it certified a fabricated binding of `ethyl`/`eth` tokens to
# N and O atoms (`1-ethylethane` for CCN(CC)N=O). No LIVE general-engine result
# exhibits this (the engine binds the atoms it perceives, so its bindings are
# element-consistent by construction), so this is defensive hardening for the T4
# tier, which ships on the certificate rather than on OPSIN-RT.
#
# The check is CONSERVATIVE and SOUND-BY-REFUSAL (mirrors name_morphemes' own
# contract, and invariant 9: a false rejection would regress breadth / unmask a
# worse generator). It rejects ONLY when a token is CONFIDENTLY all-carbon (a
# plain alkane/alkyl stem or a carbocyclic retained name) yet is bound to a
# NON-carbon heavy atom. Any token that could legitimately carry a heteroatom
# (a REPL/HW/heterocyclic/functional morpheme, or anything unrecognised) is
# SKIPPED — never rejected — so no legitimate certificate can be voided.
_CARBON_STEM = (
    r"meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec|tridec|"
    r"tetradec|pentadec|hexadec|heptadec|octadec|nonadec|icos|henicos|docos"
)
_ALL_CARBON_ALIPHATIC = re.compile(
    r"^(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?"
    r"(?:cyclo)?"
    rf"(?:{_CARBON_STEM})"
    r"(?:a)?"
    r"(?:ane|ene|yne|yl|ylidene|ylidyne|enyl|ynyl|diyl|triyl|tetrayl)?$"
)
# Pure-carbon retained substituent/parent names (no heteroatom).
_ALL_CARBON_RETAINED = frozenset({
    "phenyl", "phenylene", "benzyl", "benzylidene", "benzhydryl", "benzene",
    "toluene", "tolyl", "xylyl", "mesityl", "cumenyl", "styryl", "cinnamyl",
    "phenethyl", "trityl", "naphthyl", "naphthalenyl", "naphthalene",
    "anthryl", "anthracenyl", "anthracene", "phenanthryl", "phenanthrenyl",
    "phenanthrene", "indenyl", "indene", "indanyl", "fluorenyl", "fluorene",
    "adamantyl", "norbornyl", "azulenyl", "azulene",
})


def _token_is_confidently_all_carbon(token: str) -> bool:
    """True iff ``token`` provably declares ONLY carbon skeletal atoms.

    Normalises away locants, multipliers, stereo, enclosing marks and case, then
    matches the plain alkane/alkyl grammar or the carbocyclic retained set. Any
    heteroatom-bearing or unrecognised token returns False (so it is skipped, not
    rejected). Kept deliberately narrow: catching the alkane/alkyl class closes
    the demonstrated gap; broadening risks a false rejection.
    """
    if not token:
        return False
    t = token.strip().lower()
    # strip enclosing marks and stereo/locant noise
    t = re.sub(r"[\[\](){}]", "", t)
    t = re.sub(r"\([reszRSEZ+\-,\d\s]*\)", "", t)
    t = re.sub(r"^[0-9,''′″\-\s]+", "", t)   # leading locants
    t = re.sub(r"-[0-9,''′″]+-", "-", t)      # interior locant runs
    t = t.replace("-", "").replace(",", "").strip()
    if not t:
        return False
    if t in _ALL_CARBON_RETAINED:
        return True
    return bool(_ALL_CARBON_ALIPHATIC.match(t))


def _verify_partition(mol, name, pairs, allow_charged: bool = False,
                      atoms=None) -> E1Verdict:
    """Shape-agnostic core of E1's atom-coverage certificate.

    Extracted verbatim from the historical ``verify_certificate`` body so that
    ONE certification core is shared across every producer whose output is an
    atom->token partition (invariant 12: extend, don't duplicate). Two callers
    today: ``verify_certificate`` (the object-shape wrapper below, for a
    ``GeneralEngineResult``) and the universal recursive namer
    (``assembly/universal_substituent.py``), whose ``UniversalResult.bindings``
    is *already* ``Tuple[Tuple[str, FrozenSet[int]], ...]`` -- exactly ``pairs``.

    Args:
      * ``name``   -- the emitted name string (token-in-name check target).
      * ``pairs``  -- any iterable of ``(token, atom_ids)`` 2-tuples. Materialised
        here (``list``), so a one-shot generator is fine (iterated three times).
      * ``allow_charged`` -- lifts the G1 net-formal-charge refusal, for a caller
        that owns the charge axis by a stronger, earlier check (the universal
        namer's per-atom raw-formal-charge void guard, task B2b, is exactly that).
      * ``atoms`` -- restricts the partition's reference set to a given heavy-atom
        index set (a BRANCH subgraph). ``None`` (the whole-molecule default used
        by ``verify_certificate``) => every heavy atom of ``mol``, byte-identical
        to the historical body.

    Checks, in order: P1 atom partition (every reference-set heavy atom bound by
    exactly one token, no double-count, no phantom/non-reference atom),
    token-in-name (each non-empty token is a literal substring of ``name``), F-E1
    chemistry-soundness (a confidently all-carbon token may not bind a
    heteroatom), G1 charge scope.
    """
    pairs = list(pairs)
    if atoms is None:
        heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    else:
        heavy = set(atoms)
    seen = {}
    for token, atom_ids in pairs:
        for idx in atom_ids:
            if idx in seen:
                return E1Verdict(False, f"atom {idx} bound twice "
                                        f"({seen[idx]!r} and {token!r})")
            seen[idx] = token
    unbound = heavy - set(seen)
    if unbound:
        return E1Verdict(False, f"unbound heavy atoms: {sorted(unbound)}")
    phantom = set(seen) - heavy
    if phantom:
        return E1Verdict(False, f"bindings reference non-heavy/missing "
                                f"atoms: {sorted(phantom)}")
    for token, _atom_ids in pairs:
        if token and token not in name:
            return E1Verdict(False, f"token {token!r} not in name")
    # F-E1: a CONFIDENTLY all-carbon token may not be bound to a heteroatom.
    # Sound-by-refusal -- only fires on the alkane/alkyl/carbocyclic class; every
    # other token is skipped, so no legitimate certificate is voided.
    for token, atom_ids in pairs:
        if not token or not _token_is_confidently_all_carbon(token):
            continue
        for idx in atom_ids:
            z = mol.GetAtomWithIdx(idx).GetAtomicNum()
            if z != 6:
                sym = mol.GetAtomWithIdx(idx).GetSymbol()
                return E1Verdict(
                    False,
                    f"all-carbon token {token!r} bound to {sym} atom {idx}",
                )
    if not allow_charged and Chem.GetFormalCharge(mol) != 0:
        return E1Verdict(False, "net formal charge nonzero (G1 charge scope)")
    return E1Verdict(True, "ok")


def verify_certificate(mol, result, allow_charged: bool = False) -> E1Verdict:
    """Atom-partition certificate for a GeneralEngineResult.

    v26 P5: ``allow_charged`` (set only under ``complete``) lifts the
    net-formal-charge refusal -- the charge is expressed as a
    ``-ylium``/``-ide``/``-uide``/``-ium`` suffix on an already-bound skeletal
    atom, so it introduces NO new atom and the partition is still complete.
    Default False -> byte-identical (the G1 neutral-scope guard still fires).

    Thin object-shape wrapper over ``_verify_partition``: unpacks each
    ``result.bindings`` element's ``.token``/``.atom_ids`` into a ``(token,
    atom_ids)`` 2-tuple. Byte-identical logic to the historical body.
    """
    return _verify_partition(
        mol, result.name,
        ((b.token, b.atom_ids) for b in result.bindings),
        allow_charged=allow_charged)
