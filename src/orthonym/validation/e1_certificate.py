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
    token-in-name (each non-empty token is a substring of ``name``, tolerating
    the P-16.7.1(a)/P-74.1.1 terminal-'e' elision before a vowel-initial ionic
    suffix), F-E1 chemistry-soundness (a confidently all-carbon token may not
    bind a heteroatom), G1 charge scope.
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
    # Token-in-name: every non-empty token is a literal substring of ``name``.
    # ELISION-ROBUST on THIS axis only (P-16.7.1(a) / P-74.1.1): a parent-hydride
    # token's terminal 'e' is elided before a vowel-initial ionic/cumulative
    # suffix (``2-azapropane`` -> ``...2-azapropan-2-ium``), so the FULL token
    # is legitimately absent while its stem (token without the trailing 'e') is
    # present. Accept the stem too. This tolerance is confined to token-in-name;
    # the stored token stays the FULL form, so P1 and F-E1 both see the real,
    # classifiable token. Existing GeneralEngineResult callers are unaffected
    # (their parent tokens are already elision-robust stems, e.g. 'but'/'benzen',
    # so the extra clause never changes their verdict); the relaxation only ever
    # ACCEPTS the one legitimate elision, never admits an unrelated token.
    for token, _atom_ids in pairs:
        if not token:
            continue
        if token in name:
            continue
        if token.endswith("e") and token[:-1] in name:
            continue
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


def verify_atom_coverage(mol, name, atom_id_groups, allow_charged: bool = True):
    """Producer-side atom-COVERAGE close (the task-W2 reusable template).

    A thin wrapper over :func:`_verify_partition` for the recurring producer
    situation the W2 jar-absent-proper fix addresses: *"here are the atom groups
    my emitted ``name`` accounts for — confirm together they cover EVERY heavy
    atom of ``mol``, else I must decline (fail-closed) rather than ship a name
    that silently dropped atoms."*

    This is the OPSIN-free, jar-independent guard that catches an atom-drop at
    CONSTRUCTION (defense-in-depth: jar-present SELF-01 would also catch it, but
    only after a JVM round-trip; jar-absent nothing else does). It reuses the E1
    partition primitive rather than duplicating a coverage check (invariant 12:
    extend, don't duplicate) and is deliberately scoped to the COVERAGE axis:

    * Groups are DEDUPLICATED before binding, so a benign double-listing of an
      atom never trips the partition's ``bound twice`` refusal (a false void that
      would cost breadth — the fix must void ONLY genuine drops).
    * Each group is bound under the EMPTY token, so the token-in-name and F-E1
      chemistry checks are SKIPPED here. Those two checks presuppose a
      ``GeneralEngineResult``-style ``(token, atom_ids)`` stream whose token
      strings are reconstructable; a legacy composer producer building the
      partition from atom-index data alone cannot supply faithful token strings
      (locants/elision/enclosing-marks are added around morphemes at assembly
      time), and a mismatched token would false-void a CORRECT name. Coverage is
      the only axis this template owns.
    * ``allow_charged`` defaults True: this template checks atom coverage, not
      the G1 charge scope, so a legitimately-charged parent is never voided for a
      reason unrelated to an atom drop.

    Args:
      * ``atom_id_groups`` -- iterable of iterables of heavy-atom indices, one per
        name-piece the producer emitted (ring atoms, each named substituent's
        atoms, suffix atoms, ...). Their UNION is the accounted set.

    Returns an :class:`E1Verdict`; ``.ok`` is False (with an ``unbound heavy
    atoms`` reason) when any heavy atom is left unaccounted. Callers decline
    (return ``None`` / abstain) on a non-ok verdict.
    """
    seen: set = set()
    pairs = []
    for grp in atom_id_groups:
        g = frozenset(grp) - seen
        if g:
            pairs.append(("", g))
            seen |= g
    return _verify_partition(mol, name, pairs, allow_charged=allow_charged)


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
