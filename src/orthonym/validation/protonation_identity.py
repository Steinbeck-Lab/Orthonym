"""Protonation-site identity: does a name's OPSIN parse carry its hydrons on the
same atoms as the input structure?

The standard InChI moves every hydron of a protonated (onium) atom into one
mobile /p layer, so the full standard InChIKey cannot tell WHICH atom carries
the charge. Protonation isomers of one cation share one key:

    C[NH2+]CCC(=O)NC vs CNCCC(=O)[NH2+]C (aminium vs amidium)
    C[NH2+]CCC#N vs CNCCC#[NH+] (aminium vs nitrilium)
    C[NH+](C)CCN vs CN(C)CC[NH3+] (tertiary vs primary aminium)

Every gate that accepts a name on full-key equality alone is therefore blind to
a name that puts the '-ium' on the wrong nitrogen ('N-methyl-3-(methylamino)-
propanamidium' for the protonated amine). (the Blue Book) and
Table 7.4 (:41417-41424) define 'amidium', 'nitrilium', 'aminium' as the
cationic form OF that characteristic group, so such a name denotes a different
species.

The same holds for a DEPROTONATED site, and for the hydron of a cation whose
charge sits on a substituted atom. The standard InChI adds the missing hydron
back (or takes the extra one off) and records it in the /p layer, so these
share one key too:

    O=C([O-])Cc1ccccc1C(=O)O.[Na+] vs O=C(O)Cc1ccccc1C(=O)[O-].[Na+]
    C[N+]1=CNc2ccccc21 vs C[NH+]1C=Nc2ccccc21

'sodium 2-(carboxymethyl)benzoate' is the second salt (the ring carboxylate is
the anion); the first is 'sodium (2-carboxyphenyl)acetate'. "Acid
salts" (the Blue Book-31596): method (1) cites "the free acid... as a
prefix to the name of the anion", and an anion is senior to an acid, '4 Anions' before '7 Acids',:18167). '1-methyl-1H-benzimidazol-1-ium'
puts the added hydron on N-1, beside the methyl group; the first cation, with the
hydron on N-3, is '1-methyl-1H-benzimidazol-3-ium',:41368: the
'ium' cation is formed "by adding one or more hydrons to any position"; the
example '1H-imidazol-3-ium (PIN)',:41396).

Two rules, applied in order:

1. When BOTH the input and the parse carry a protonated heavy atom (a
   positively charged non-hydrogen atom bearing hydrogen), the fixed-hydrogen
   InChI (``/FixedH``, stereo layers off) must be equal. The fixed-H layer
   records where each hydron sits, and InChI keeps charge-delocalised forms
   equal (the two resonance drawings of an amidinium or a 4-aminopyridinium give
   one fixed-H InChI), so a correct name is never rejected for drawing the
   charge on the other resonance atom.

2. Otherwise, when both structures carry a charged atom, the input carries no
   bare proton ([H+], whose site the input leaves open) and the standard InChIs
   (stereo off) are equal -- the case the standard key cannot see -- unequal
   fixed-H InChIs are a mismatch unless one of three things explains the
   difference (the rule reads the name only in (c)):
   (a) a neutral tautomer only: the structures' neutral analogues
       (``_neutral_analogue_inchi``: each charged atom replaced by its
       isoelectronic neutral atom, in any resonance drawing) share one standard
       InChI, so the charges sit on the same atoms and only neutral hydrons moved
       inside one mobile group
       (a 2-pyridone drawn as the 2-hydroxypyridine beside a nitro group) -- the
       tautomer tolerance every gate keeps for a neutral molecule;
   (b) the amino-acid zwitterion drawn in the neutral form: the parse is the
       input with ONE alpha-amino-acid pair neutralised, the hydron moved from the
       ammonium N to the carboxylate of the same R-CH(NH3+)-COO- unit
       (``_neutralised_pair_matches``). This is the part-molecule case of the
       zwitterion that (the Blue Book) names as the neutral
       amino acid ('2-aminopropanoic acid rather than... 2-azaniumylpropanoate');
       another pairing (the ammonium with a side-chain carboxylate or a sulfonate)
       is another protomer.
   (c) only for a name that cites the acid hydrons with the method (2) word
       'hydrogen' / 'dihydrogen' in the salt position, between the cation and
       the anion words (``_names_acid_hydrons_without_site``): the hydrons of
       noncarbon oxoacid groups sit on other acid chalcogens of EQUIVALENT
       central atoms of ONE connected component (``_oxoacid_centre_hydron_key``).
       A name that multiplies an ion word citing hydrogen ('disodium bis(methyl
       hydrogen phosphate)') names identical units: when the input is those
       identical units as drawn, their hydrons are pooled across the units
       (``_names_multiplied_hydrogen_unit``, ``_oxoacid_fragments_identical``).
       (the Blue Book) names the acid salts of di- and polynuclear noncarbon
       oxoacids "in the same way as neutral salts, the remaining acid hydrogen
       atom(s) being indicated by the word 'hydrogen' (or 'dihydrogen', etc., as
       appropriate)", and (:31619) the acid salts of organic
       derivatives of polybasic inorganic oxoacids by the same method (2); such a
       name does not place the hydrons among equivalent groups ('disodium
       dihydrogen diphosphate' and 'disodium dihydrogen methanediphosphonate' are
       read by OPSIN with both hydrons on one phosphorus). A hydron on a centre
       of another kind (the ester phosphorus of a methyl diphosphate, the sulfur
       of a sulfuric-phosphoric anhydride), or a substitutive name that places the
       hydrons ('(dichloro-phosphonomethyl)phosphonate'), stays a mismatch.
   A hydron (or a charge) that moves to ANOTHER acid or base site is a
   different species.

Scoped on purpose: a salt drawn as its ionic pair and named as the neutral
acid-base form ('...amine hydrochloride', a proton on Cl), or an amino-acid
zwitterion named as the neutral amino acid, has a parse with no charged atom and
is not a protonation-SITE question; it keeps its existing treatment, judged by
the callers' own charge checks.
"""
import re

from rdkit import Chem
from rdkit.Chem import inchi as _inchi


def _has_protonated_heavy_atom(mol) -> bool:
    return any(a.GetAtomicNum() > 1 and a.GetFormalCharge() > 0 and a.GetTotalNumHs() > 0
               for a in mol.GetAtoms())


def _has_charged_atom(mol) -> bool:
    return any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())


def _has_bare_proton(mol) -> bool:
    return any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 and a.GetFormalCharge() > 0
               for a in mol.GetAtoms())


def _fixed_h_inchi(mol):
    try:
        return _inchi.MolToInchi(mol, options="/FixedH /SNon")
    except Exception:
        return None


def _standard_inchi(mol):
    try:
        return _inchi.MolToInchi(mol, options="/SNon") or None
    except Exception:
        return None


#: The neutral atom isoelectronic with a singly charged one (same valence
#: electron count, so the same bonds and hydrogens), for ``_neutral_analogue_inchi``.
_ISOELECTRONIC_NEUTRAL = {
    (6, 1): 5, (7, 1): 6, (8, 1): 7, (15, 1): 14, (16, 1): 15, (33, 1): 32,
    (34, 1): 33, (35, 1): 34, (52, 1): 51, (53, 1): 52,
    (5, -1): 6, (6, -1): 7, (7, -1): 8, (8, -1): 9, (13, -1): 14, (14, -1): 15,
    (15, -1): 16, (16, -1): 17, (34, -1): 35, (52, -1): 53,
}


def _neutral_analogue_inchi(mol):
    """Standard InChI (stereo off) of ``mol`` with every singly charged main-group
    atom replaced by its isoelectronic neutral atom (N+ -> C, O- -> F,...) keeping
    its bonds and hydrogens, and every unbonded charged atom without hydrogen (a
    metal cation, a halide) left out. In the analogue the charges are fixed atoms,
    so its mobile-H groups hold only the NEUTRAL heteroatoms' hydrons: two
    structures whose analogues share one standard InChI carry their charges on
    the same atoms and differ at most by a neutral tautomer. None when an atom
    has no entry (a charge of 2 or more, a bonded charged metal, a bare proton)
    or the analogue does not sanitize."""
    rw = Chem.RWMol(mol)
    drop = []
    for atom in rw.GetAtoms():
        q = atom.GetFormalCharge()
        if not q:
            continue
        if atom.GetAtomicNum() == 1:
            return None
        if atom.GetDegree() == 0 and atom.GetTotalNumHs() == 0:
            drop.append(atom.GetIdx())
            continue
        z = _ISOELECTRONIC_NEUTRAL.get((atom.GetAtomicNum(), q))
        if z is None:
            return None
        n_h = atom.GetTotalNumHs()
        atom.SetAtomicNum(z)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(n_h)
    for idx in sorted(drop, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        return _inchi.MolToInchi(out, options="/SNon") or None
    except Exception:
        return None


#: Resonance structures enumerated per side for ``_same_up_to_neutral_tautomer``.
_MAX_RESONANCE_STRUCTS = 64


def _analogue_inchis(mol) -> set:
    """``_neutral_analogue_inchi`` of ``mol`` and of each of its resonance
    structures: a delocalised charge is drawn on either atom (the two nitrogens of
    an imidazolium), and each drawing gives its own analogue."""
    out = set()
    a = _neutral_analogue_inchi(mol)
    if a:
        out.add(a)
    try:
        sup = Chem.ResonanceMolSupplier(mol, 0, _MAX_RESONANCE_STRUCTS)
        for i in range(len(sup)):
            m = sup[i]
            if m is not None:
                a = _neutral_analogue_inchi(m)
                if a:
                    out.add(a)
    except Exception:
        pass
    return out


def _same_up_to_neutral_tautomer(mi, mo) -> bool:
    """True iff some drawing of each structure has the same neutral analogue."""
    return bool(_analogue_inchis(mi) & _analogue_inchis(mo))


def _neutralised(mol, atom_idxs):
    """``mol`` with each atom of ``atom_idxs`` made neutral by one hydron (an
    onium atom loses one, an anionic atom gains one); None if it does not
    sanitize."""
    rw = Chem.RWMol(mol)
    for idx in atom_idxs:
        atom = rw.GetAtomWithIdx(idx)
        n_h = atom.GetTotalNumHs()
        q = atom.GetFormalCharge()
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(n_h - 1 if q > 0 else n_h + 1)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def _amino_acid_zwitterion_pairs(mol):
    """(onium N, carboxylate O-) pairs of the alpha-amino-acid units of ``mol``:
    an ammonium nitrogen bearing hydrogen on the carbon that carries a -C(=O)O(-)
    group, R-CH(NH3+)-COO-."""
    pairs = []
    for n in mol.GetAtoms():
        if n.GetAtomicNum() != 7 or n.GetFormalCharge() != 1 or n.GetTotalNumHs() == 0:
            continue
        for ca in n.GetNeighbors():
            if ca.GetAtomicNum() != 6:
                continue
            for cc in ca.GetNeighbors():
                if cc.GetAtomicNum() != 6 or cc.GetIdx() == n.GetIdx():
                    continue
                oxy = [o for o in cc.GetNeighbors() if o.GetAtomicNum() == 8]
                if len(oxy) != 2 or cc.GetDegree() != 3:
                    continue
                for o in oxy:
                    if o.GetFormalCharge() == -1 and o.GetDegree() == 1:
                        pairs.append((n.GetIdx(), o.GetIdx()))
    return pairs


def _neutralised_pair_matches(mi, mo, fixed_h_parse) -> bool:
    """True iff neutralising ONE alpha-amino-acid zwitterion pair of the input
    (the ammonium N and the carboxylate of the same amino-acid unit) gives the
    parse, fixed-H exact or up to a neutral tautomer (rule 2b of the module
    docstring)."""
    for pair in _amino_acid_zwitterion_pairs(mi):
        m = _neutralised(mi, pair)
        if m is None:
            continue
        if _fixed_h_inchi(m) == fixed_h_parse or _same_up_to_neutral_tautomer(m, mo):
            return True
    return False


#: Central atoms of the noncarbon oxoacids whose acid salts the Blue Book names
#: with a separate 'hydrogen' word whatever the hydron's site,
#:, and the chalcogens of their acid groups (-OH / -O(-) and their
#: chalcogen analogues).
_OXOACID_CENTRES = frozenset({5, 14, 15, 16, 33, 34, 51, 52})
_ACID_CHALCOGENS = frozenset({8, 16, 34, 52})

#: The method (2) acid-salt word: 'hydrogen', 'dihydrogen',...
_HYDROGEN_WORD_RE = re.compile(r"^(?:di|tri|tetra|penta|hexa|hepta|octa)?hydrogen$")
#: A multiplied ion word whose unit cites acid hydrogen: 'bis(methyl hydrogen phosphate)'.
_MULTIPLIED_UNIT_RE = re.compile(r"^(?:bis|tris|tetrakis)\((?P<unit>.+)\)$")


def _top_level_words(name: str):
    """The space-separated words of ``name`` outside any enclosing marks."""
    words, depth, cur = [], 0, ''
    for ch in name:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ' ' and depth == 0:
            if cur:
                words.append(cur)
            cur = ''
        else:
            cur += ch
    if cur:
        words.append(cur)
    return words


def _names_acid_hydrons_without_site(name) -> bool:
    """True iff ``name`` cites acid hydrons with the method (2) word in the salt
    position: a 'hydrogen' / 'dihydrogen'... word outside enclosing marks,
    after at least one word (the cation or cations, (2) "inserted as
    a separate word between the name(s) of the cation(s) and the name of the
    anion", the Blue Book) and before the anion word."""
    words = _top_level_words(name) if name else []
    return any(_HYDROGEN_WORD_RE.match(w) for w in words[1:-1])


def _names_multiplied_hydrogen_unit(name) -> bool:
    """True iff ``name`` multiplies an ion word that cites acid hydrogen:
    'disodium bis(methyl hydrogen phosphate)' -- identical units, each with its
    own hydron."""
    for w in (_top_level_words(name) if name else [])[1:]:
        m = _MULTIPLIED_UNIT_RE.match(w)
        if m and any(_HYDROGEN_WORD_RE.match(u) for u in _top_level_words(m.group('unit'))):
            return True
    return False


def _oxoacid_centre_hydron_key(mol, across_fragments: bool = False):
    """Canonical SMILES of ``mol`` with the hydrons of the noncarbon oxoacid
    groups pooled per set of EQUIVALENT central atoms: every terminal acid
    chalcogen of such a centre loses its hydrogens, charge and double bond, and
    one dummy atom bonded to all centres of a symmetry class carries the class's
    hydron count and charge. Two structures with the same key differ only by where
    the hydrons sit among equivalent acid groups (the two phosphorus atoms of a
    diphosphate or of a methanediphosphonate); a hydron on a centre of ANOTHER
    class (the ester phosphorus of a methyl diphosphate, the sulfur of a
    sulfuric-phosphoric anhydride) changes the key. None when ``mol`` has fewer
    than two equivalent such centres."""
    centres = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetAtomicNum() in _OXOACID_CENTRES
               and any(nb.GetAtomicNum() in _ACID_CHALCOGENS and nb.GetDegree() == 1
                       for nb in a.GetNeighbors())]
    if len(centres) < 2:
        return None
    rw = Chem.RWMol(mol)
    terminal = {}
    for c in centres:
        terminal[c] = []
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            if nb.GetAtomicNum() in _ACID_CHALCOGENS and nb.GetDegree() == 1:
                terminal[c].append(nb.GetIdx())
                a = rw.GetAtomWithIdx(nb.GetIdx())
                a.SetFormalCharge(0)
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
                rw.GetBondBetweenAtoms(c, nb.GetIdx()).SetBondType(Chem.BondType.SINGLE)
    stripped = rw.GetMol()
    try:
        stripped.UpdatePropertyCache(strict=False)
        ranks = list(Chem.CanonicalRankAtoms(stripped, breakTies=False))
    except Exception:
        return None
    frag_of = {}
    for fi, idxs in enumerate(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)):
        for i in idxs:
            frag_of[i] = fi
    classes = {}
    for c in centres:
        # One connected component unless identical units are pooled (see the
        # caller): speaks of one di- or polynuclear acid.
        cls = ranks[c] if across_fragments else (ranks[c], frag_of[c])
        classes.setdefault(cls, []).append(c)
    if not any(len(v) >= 2 for v in classes.values()):
        return None
    for cls in classes.values():
        n_h = sum(mol.GetAtomWithIdx(t).GetTotalNumHs() for c in cls for t in terminal[c])
        charge = sum(mol.GetAtomWithIdx(t).GetFormalCharge() for c in cls for t in terminal[c])
        d = Chem.Atom(0)
        d.SetIsotope(n_h + 1)
        d.SetFormalCharge(charge)
        d.SetNoImplicit(True)
        di = rw.AddAtom(d)
        for c in cls:
            rw.AddBond(di, c, Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        out.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(out, allHsExplicit=True)
    except Exception:
        return None


def _same_oxoacid_hydrons_among_equivalent_centres(mi, mo, across_fragments=False) -> bool:
    ki = _oxoacid_centre_hydron_key(mi, across_fragments)
    return ki is not None and ki == _oxoacid_centre_hydron_key(mo, across_fragments)


def _oxoacid_fragments_identical(mol) -> bool:
    """True iff every fragment of ``mol`` that carries a noncarbon oxoacid centre
    is the same structure as drawn (identical units, hydrons included), and
    there are at least two such fragments."""
    frags = []
    for f in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        if any(a.GetAtomicNum() in _OXOACID_CENTRES and any(
                nb.GetAtomicNum() in _ACID_CHALCOGENS and nb.GetDegree() == 1
                for nb in a.GetNeighbors()) for a in f.GetAtoms()):
            frags.append(Chem.MolToSmiles(f))
    return len(frags) >= 2 and len(set(frags)) == 1


def protonation_site_verdict(input_smiles: str, parsed_smiles: str,
                             name: str = None) -> str:
    """``"n/a"``, ``"ok"`` or ``"mismatch"``.

    Both structures carry a protonated heavy atom: ``"ok"`` if the fixed-H
    InChIs are equal, else ``"mismatch"`` (rule 1 of the module docstring).
    Both carry a charged atom, the input has no bare proton and the standard
    InChIs (stereo off) are equal: ``"mismatch"`` if the fixed-H InChIs differ
    and neither a neutral tautomer, the neutralised amino-acid zwitterion pair
    nor -- for a ``name`` with the method (2) 'hydrogen' word -- the hydrons of
    equivalent oxoacid centres explain it (rule 2), else ``"n/a"``. Anything
    else is ``"n/a"``. ``name`` is the name whose parse ``parsed_smiles`` is;
    without it rule 2 (c) does not apply. An input or parse RDKit cannot read, or a fixed-H InChI
    that cannot be computed, is ``"n/a"``: the callers' own checks decide
    those."""
    mi = Chem.MolFromSmiles(input_smiles) if input_smiles else None
    mo = Chem.MolFromSmiles(parsed_smiles) if parsed_smiles else None
    if mi is None or mo is None:
        return "n/a"
    if _has_protonated_heavy_atom(mi) and _has_protonated_heavy_atom(mo):
        fi, fo = _fixed_h_inchi(mi), _fixed_h_inchi(mo)
        if not fi or not fo:
            return "n/a"
        return "ok" if fi == fo else "mismatch"
    if not (_has_charged_atom(mi) and _has_charged_atom(mo)) or _has_bare_proton(mi):
        return "n/a"
    if _standard_inchi(mi) != _standard_inchi(mo):
        return "n/a"   # the standard InChI already tells them apart
    fi, fo = _fixed_h_inchi(mi), _fixed_h_inchi(mo)
    if not fi or not fo or fi == fo:
        return "n/a"
    if _same_up_to_neutral_tautomer(mi, mo) or _neutralised_pair_matches(mi, mo, fo):
        return "n/a"
    if (_names_acid_hydrons_without_site(name)
            and _same_oxoacid_hydrons_among_equivalent_centres(mi, mo)):
        return "n/a"
    # 'bis(<unit with hydrogen>)' names identical units: the input must be those
    # identical units as drawn; the reading may put their hydrons on either one.
    if (_names_multiplied_hydrogen_unit(name) and _oxoacid_fragments_identical(mi)
            and _same_oxoacid_hydrons_among_equivalent_centres(mi, mo, across_fragments=True)):
        return "n/a"
    return "mismatch"
