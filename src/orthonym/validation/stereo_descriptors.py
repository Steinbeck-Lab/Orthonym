"""The descriptor check of a name for its input: is every CIP stereodescriptor right?

A name that OPSIN reads only without its stereodescriptors (a lowercase ``r``/``s``, a
descriptor OPSIN 2.9.0 does not know) is shipped by the default tier's stereo-layer carve-out
after the stereo-stripped parse proved the constitution , and the best-effort tier's
cis/trans offer starts from the same name. Neither may ship a name whose descriptors are
wrong, so both ask:func:`stereo_block_units` for the reading by which the name describes the
input. ``orthonym.namer`` imports:func:`stereo_block_units` and
:func:`stereo_descriptors_verified` as ``_stereo_block_units`` and
``_stereo_descriptors_verified``; its callers resolve those names at call time.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem



#: One CIP descriptor of a stereodescriptor block: an optional locant (digits, an
#: optional letter like '4a', primes) and the code. The locant is absent only
#: where the stereogenic unit has none: "preceded by a numerical or letter
#: locant... when such locants are present"), e.g. the sulfur of
#: 'ethyl (R)-4-nitrobenzene-1-sulfinate'.
_CIP_DESCRIPTOR_TOKEN_RE = re.compile(r"^(\d+[a-z]?['′]*)?([RSrsEZ])$")


def _cip_descriptor_tokens(blocks) -> Optional[list]:
    """One list of ``(locant, code)`` per descriptor block of ``blocks`` (the blocks
    ``rules.stereochemistry.strip_stereo_blocks`` removed, in that order), primes
    dropped from the locant (``None`` for a locant-free code); ``None`` when a block
    holds anything else (a relative-configuration word such as 'rel' / 'cis', 'R*',
    'RS',...): such a descriptor does not assert one configuration of one stereogenic
    unit, so it cannot be checked."""
    by_block = []
    for block in blocks:
        if not (block.startswith("(") and block.endswith(")")):
            return None
        tokens = []
        for tok in block[1:-1].split(","):
            m = _CIP_DESCRIPTOR_TOKEN_RE.match(tok.strip())
            if not m:
                return None
            loc = m.group(1)
            tokens.append((loc.replace("'", "").replace("′", "") if loc else None,
                           m.group(2)))
        by_block.append(tokens)
    return by_block


def _bare_skeleton(mol):
    """A copy of ``mol`` with the same atom and bond indices, every bond single and
    every atom neutral, non-aromatic and hydrogen-free: the graph two tautomers,
    two Kekulé structures or two charge forms of one constitution share."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
        b.SetStereo(Chem.BondStereo.STEREONONE)
    rw.UpdatePropertyCache(strict=False)
    return rw


@dataclass(frozen=True)
class StereoBlockUnits:
    """One descriptor block of a name and what the accepted reading of it binds
    (:func:`stereo_block_units`).

    ``text``: the block as ``rules.stereochemistry.strip_stereo_blocks`` returns it
    (``'(1r,4r)'``). ``span``: ``(start, end)`` in the name, the block with its
    following hyphen (``name[start:end] == '(1r,4r)-'``). ``multiplicity``: how many
    copies of its group the name cites the block for (1 outside a multiplied group).
    ``units``: the stereo units of the input the reading binds to the tokens of the
    block, ``multiplicity`` per token, as ``('a', atom_idx)`` (a stereocentre) or
    ``('b', bond_idx)`` (a double bond), in ascending order. ``token_units``: the same units
    per token of the block, in the order of its tokens (``multiplicity`` each)."""
    text: str
    span: Tuple[int, int]
    multiplicity: int
    units: Tuple[Tuple[str, int], ...]
    token_units: Tuple[Tuple[Tuple[str, int], ...], ...] = ()


#: ``(word, count)`` of every multiplying prefix, longest word first (set on first use).
_BLOCK_MULTIPLIERS: Optional[Tuple[Tuple[str, int], ...]] = None


def _block_multiplier_words() -> Tuple[Tuple[str, int], ...]:
    """The multiplying prefixes of ``data.multipliers`` (``di``... ``icosa`` and
    ``bis``... ``icosakis``, as ``(word, count)``, the longest word first."""
    global _BLOCK_MULTIPLIERS
    if _BLOCK_MULTIPLIERS is None:
        from ..data.multipliers import COMPLEX_MULTIPLIERS, SIMPLE_MULTIPLIERS
        words = {word: n for table in (SIMPLE_MULTIPLIERS, COMPLEX_MULTIPLIERS)
                 for n, word in table.items()}
        _BLOCK_MULTIPLIERS = tuple(sorted(words.items(),
                                          key=lambda kv: (-len(kv[0]), kv[0])))
    return _BLOCK_MULTIPLIERS


def _block_multiplicity(name: str, start: int) -> int:
    """How many copies of its group a descriptor block that starts at ``start`` in
    ``name`` is cited for: the product of the multiplying prefixes (``bis``, ``tris``,
    ... or ``di``, ``tri``,... written directly before the opening mark) of every
    enclosing group. ``bis[(1r,4r)-4-methylcyclohexyl]phosphane`` (the Blue Book,
     "Achiral cyclic compounds") cites ``(1r,4r)`` for both rings; a block
    outside every multiplied group is cited once.

    The groups are found with ``assembly.naming_utils.enclosing_opener``; the block's
    own parentheses are not an enclosing group (``start`` is the block's first
    character)."""
    from ..assembly.naming_utils import enclosing_opener
    words = _block_multiplier_words()
    multiplicity, pos = 1, start
    while True:
        opener = enclosing_opener(name, pos)
        if opener is None:
            return multiplicity
        head = name[:opener]
        for word, n in words:
            if head.endswith(word):
                multiplicity *= n
                break
        pos = opener


def _stereo_unit_given_by_parse(mol, omol, inv, unit) -> bool:
    """Does the OPSIN parse ``omol`` of the stereo-stripped name already specify the
    unit ``unit`` of ``mol`` (a retained parent's own configuration, a 17β,...)?
    ``inv`` maps an atom index of ``mol`` to the index of its atom in ``omol``."""
    kind, idx = unit
    if kind == 'a':
        return (omol.GetAtomWithIdx(inv[idx]).GetChiralTag()
                != Chem.ChiralType.CHI_UNSPECIFIED)
    b = mol.GetBondWithIdx(idx)
    ob = omol.GetBondBetweenAtoms(inv[b.GetBeginAtomIdx()], inv[b.GetEndAtomIdx()])
    return ob is not None and ob.GetStereo() != Chem.BondStereo.STEREONONE


def stereo_block_units(name: str, smiles: Optional[str],
                        reduced: str) -> Optional[List[StereoBlockUnits]]:
    """The reading by which ``name`` describes the stereo of ``smiles``, or ``None``
    when there is none. One:class:`StereoBlockUnits` per descriptor block of
    ``name``, in the order ``rules.stereochemistry.strip_stereo_blocks`` removes them;
    the stereo-layer carve-out and the best-effort prefix-order fallback ask only
    whether the reading exists (:func:`stereo_descriptors_verified`), and a caller
    that rewrites a block reads its span and its atoms here, from the reading that was
    accepted.

    critic-1 / R63 (claims conformance, 2026-09-27): is every stereodescriptor of
    ``name`` right, for a name OPSIN rejects only for its stereo layer?

    The stereo-layer carve-out of the validity gate ships such a name after
    judged the stereo-STRIPPED parse (``reduced``, ``rules.stereochemistry.strip_stereo``), so the
    constitution is OPSIN-checked and the descriptors are not. They were shipped
    unchecked, and some were wrong: '(7R,8S)-7-chloro-1-(3,5-dihydroxyphenyl)-2-
    methylundecyl acetate' for an input whose stereocentres are the name's C1 and C2
    (C7 is unspecified, C8 is a CH2), and '(2S,3R)-3-oxoestr-4-en-17β-yl
    3-phenylpropanoate', whose C2 is a CH2 and C3 the ketone carbon.
    "NAMING OF STEREOISOMERS" (the Blue Book): "They are preceded by a
    numerical or letter locant to describe the position of the stereogenic unit
    when such locants are present" -- a descriptor asserts the configuration of the
    unit at its locant, so a descriptor on a CH2 is a wrong name, not a narrower
    OPSIN grammar.

    Generalises ``namer._pseudoasymmetric_name_verified`` (r/s only) to every CIP
    descriptor. A reading exists iff, for some reading of ``reduced`` onto the input:

    1. every descriptor block is a list of CIP code tokens (R, S, r, s, E, Z),
       each with its locant where the unit has one; anything else (``rel``,
       ``cis``, ``RS``, ``R*``) is not verifiable;
    2. every token lands on DISTINCT stereogenic units of the input whose CIP
       label (``perception.stereo.assign_stereochemistry``, the centres engine) is
       that code -- a stereocentre for R/S/r/s, a double bond at that atom for
       E/Z. A located token is read in OPSIN's own numbering of ``reduced`` (the
       ``$_AV`` atom labels of ``-o extendedsmi``), so a locant on a CH2, a
       heteroatom or out of range finds no such unit; a locant-free token may be
       any unit with its code. A block cited under a multiplier (``bis[...]``,
       ``tris[...]``) is cited for every copy of its group: each of its tokens
       lands on as many distinct units as the multiplicity
       (:func:`_block_multiplicity`), one per copy;
    3. every stereo unit of the input that the CIP labelling labels (the units a
       token can name, as in 2.) is either one of those cited units or one that the
       OPSIN parse of ``reduced`` already specifies (the retained parent's implied
       configuration, a 17β,...): a name that cites a block for fewer copies than
       the input holds leaves a labelled unit uncited and has no reading;
    4. the input with exactly the cited units made unspecified has the FULL
       InChIKey of ``reduced``'s OPSIN parse -- so each unit the parse specifies is
       what the rest of the name says. A token that restates a unit the parent
       already fixes (tropane's 1R,5S) is kept specified on both sides.

    Conditions 2 and 3 fix every labelled unit of the input, and condition 4 the
    constitution, the parent's units and any other stereo the InChI sees, so the cited
    codes fix the units they name and the name denotes the input. Condition 3 is
    stated on the units, not through the InChI of the reduced input: that InChI drops
    a lone ring centre whose partner is unspecified, so a leftover centre of a pair of
    ring centres would not show in the key. A tagged atom that carries no CIP label
    (one of the two ring centres of cis,trans-1,3,5-trimethylcyclohexane that are
    not the pseudoasymmetric one) is not a unit a descriptor can name, and is left
    to condition 4. Fails closed (``None``) on anything it cannot establish.
    """
    if not name or not smiles or not reduced:
        return None
    try:
        import itertools
        from ..namer import _self_consistency_full_key   # call time: a patch of it is seen
        from ..rules.stereochemistry import locate_stereo_blocks, strip_stereo_blocks
        stripped, blocks = strip_stereo_blocks(name)
        if stripped != reduced:
            return None
        by_block = _cip_descriptor_tokens(blocks)
        if not by_block:
            return None
        located = locate_stereo_blocks(name)
        if ([text for _s, _e, text in located] != blocks
                or any(name[s:e] != text + "-" for s, e, text in located)):
            return None
        multiplicity = [_block_multiplicity(name, s) for s, _e, _t in located]
        # (locant, code, multiplicity, index of its block), in the order of the name
        tokens = [(loc, code, multiplicity[bi], bi)
                  for bi, block_tokens in enumerate(by_block)
                  for loc, code in block_tokens]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from ..perception.stereo import assign_stereochemistry
        assign_stereochemistry(mol)
        atom_code = {a.GetIdx(): a.GetProp('_CIPCode') for a in mol.GetAtoms()
                     if a.HasProp('_CIPCode')
                     and a.GetProp('_CIPCode') in ('R', 'S', 'r', 's')}
        bond_code = {b.GetIdx(): b.GetProp('_CIPCode') for b in mol.GetBonds()
                     if b.GetBondType() == Chem.BondType.DOUBLE
                     and b.HasProp('_CIPCode') and b.GetProp('_CIPCode') in ('E', 'Z')}
        from .opsin_roundtrip import _AV_LINE_RE, _extended_smiles
        ext = _extended_smiles(reduced)
        if not ext:
            return None
        m = _AV_LINE_RE.match(ext.strip())
        if not m:
            return None
        labels = [t.strip().replace("'", "").replace('′', '')
                  for t in m.group(2).split(';')]
        omol = Chem.MolFromSmiles(m.group(1))
        if (omol is None or omol.GetNumAtoms() != len(labels)
                or omol.GetNumAtoms() != mol.GetNumAtoms()):
            return None
        reduced_key = _self_consistency_full_key(m.group(1))
        if not reduced_key:
            return None
        n = omol.GetNumAtoms()
        labelled = ({('a', i) for i in atom_code}
                    | {('b', i) for i in bond_code})
        matches = [mt for mt in mol.GetSubstructMatches(
            omol, useChirality=False, uniquify=False, maxMatches=10_000)
            if len(mt) == n]
        if not matches:
            # A tautomer, Kekulé or charge-form difference between the parse and
            # the input (the constitution itself is already verified): map the
            # bare graphs, which keep the atom indices.
            matches = [mt for mt in _bare_skeleton(mol).GetSubstructMatches(
                _bare_skeleton(omol), useChirality=False, uniquify=False,
                maxMatches=10_000) if len(mt) == n]
        by_label: Dict[str, List[int]] = {}
        for oi, lab in enumerate(labels):
            by_label.setdefault(lab, []).append(oi)
        tried: Dict[frozenset, bool] = {}
        for mt in matches:
            inv = {mi: oi for oi, mi in enumerate(mt)}
            given = {u for u in labelled
                     if _stereo_unit_given_by_parse(mol, omol, inv, u)}
            # per token, the ways to choose its units: ``multiplicity`` distinct
            # units among those at its locant that carry its code
            options = []
            for loc, code, mult, _bi in tokens:
                c = []
                for oi in (by_label.get(loc, ()) if loc is not None
                           else range(n)):
                    mi = mt[oi]
                    if code in ('E', 'Z'):
                        c.extend(('b', b.GetIdx())
                                 for b in mol.GetAtomWithIdx(mi).GetBonds()
                                 if bond_code.get(b.GetIdx()) == code)
                    elif atom_code.get(mi) == code:
                        c.append(('a', mi))
                ways = list(itertools.islice(
                    itertools.combinations(dict.fromkeys(c), mult), 64))
                if not ways:
                    break
                options.append(ways)
            else:
                for combo in itertools.islice(itertools.product(*options), 64):
                    cited = [u for part in combo for u in part]
                    if len(set(cited)) != len(cited):
                        continue
                    unspecify = [u for u in cited if u not in given]
                    # condition 3: no other labelled unit of the input is left uncited
                    if not labelled.difference(unspecify) <= given:
                        continue
                    key = frozenset(unspecify)
                    if key not in tried:
                        if len(tried) >= 128:
                            return None
                        rw = Chem.RWMol(mol)
                        for kind, idx in unspecify:
                            if kind == 'a':
                                rw.GetAtomWithIdx(idx).SetChiralTag(
                                    Chem.ChiralType.CHI_UNSPECIFIED)
                            else:
                                rw.GetBondWithIdx(idx).SetStereo(
                                    Chem.BondStereo.STEREONONE)
                        tried[key] = (_self_consistency_full_key(
                            Chem.MolToSmiles(rw)) == reduced_key)
                    if tried[key]:
                        block_units: List[List[tuple]] = [[] for _ in located]
                        for (_l, _c, _m, bi), part in zip(tokens, combo):
                            block_units[bi].append(tuple(part))
                        return [StereoBlockUnits(
                                    text=text, span=(s, e),
                                    multiplicity=multiplicity[bi],
                                    units=tuple(sorted(u for part in block_units[bi] for u in part)),
                                    token_units=tuple(block_units[bi]))
                                for bi, (s, e, text) in enumerate(located)]
        return None
    except Exception:
        return None


def stereo_descriptors_verified(name: str, smiles: Optional[str],
                                 reduced: str) -> bool:
    """Is every stereodescriptor of ``name`` right for ``smiles``, ``reduced`` being
    the name with its descriptor blocks stripped? True iff:func:`stereo_block_units`
    finds a reading; the conditions are stated there."""
    return stereo_block_units(name, smiles, reduced) is not None
