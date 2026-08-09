"""Added-carbon multi-suffix parent naming (DD1 Fix 2, Blue Book P-65.1.1.1 /
P-66.1.1.1.1.2 / P-66.5.1.1.2).

When **three or more** carboxylic-acid / carboxamide / carbonitrile groups are
present on an acyclic skeleton they cannot all be expressed as chain-terminal
``-oic acid`` / ``-amide`` / ``-nitrile`` suffixes (a chain has only two ends),
so the PIN names the **parent hydride formed by removing the carbonyl/nitrile
carbons** and appends a multiplied ``carbo*`` *added-carbon* suffix, citing one
locant per attachment atom::

    NC(=O)C(C(=O)N)C(=O)N      -> methanetricarboxamide
    CCCC(C#N)(C#N)C#N          -> butane-1,1,1-tricarbonitrile
    OC(=O)CC(C(=O)O)CC(=O)O    -> propane-1,2,3-tricarboxylic acid

Why ``n >= 3`` and NOT geminal-2: two such groups can ALWAYS be routed as the two
termini of a single chain (``HOOC-CH2-COOH`` -> propanedioic acid, the carbonyl
carbons + the bridging carbon form a 3-atom chain), so the chain-suffix form wins
for n<=2 (P-65.1.1.2). Only at n>=3 does the chain run out of ends, forcing the
added-carbon form that expresses the maximum number of groups as the suffix
(P-65.1.1.1: maximum number of skeletal/principal groups expressed as suffix).

The namer is deliberately SCOPED to a clean acyclic carbon parent (every parent
carbon's only non-chain heavy neighbours are the added carbons). A decorated
parent (extra substituents, heteroatoms, unsaturation, rings) returns ``None`` so
the caller falls through to the legacy path — never a wrong name. Ring-attached
added-carbon suffixes (``cyclohexane-1,2-dicarboxylic acid``) are already handled
by the ring suffix path and are intentionally excluded here.
"""
from __future__ import annotations

from typing import Any, List, Optional

from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
from ..data.chain_names import get_chain_prefix
from ..perception.chains import find_longest_carbon_chain
from .seniority import SUFFIX_FORMS

# Principal-group FG names that take an added-carbon ``carbo*`` suffix form and
# whose match tuple carries the carbonyl/nitrile carbon at index 0. The value is
# the added-carbon (ring/multi) suffix string from SUFFIX_FORMS[name][1].
_ADDED_CARBON_FGS = ("carboxylic_acid", "primary_amide", "nitrile")


def _added_carbon_suffix(fg_name: str) -> Optional[str]:
    forms = SUFFIX_FORMS.get(fg_name)
    if not forms:
        return None
    return forms[1]  # the ring/added-carbon ('carbo*') form


def requires_added_carbon_suffix(mol, fg_name: Optional[str], pg_matches) -> bool:
    """Cheap structural gate: principal group is acid/amide/nitrile and there are
    >= 3 distinct instances. The rigorous acyclic-clean-parent validation lives in
    :func:`name_added_carbon_parent`, which returns ``None`` to fall through.
    """
    if fg_name not in _ADDED_CARBON_FGS:
        return False
    if not pg_matches:
        return False
    added = {m[0] for m in pg_matches if m}
    return len(added) >= 3


def _added_carbons_and_attachments(mol, pg_matches):
    """Return (sorted added-carbon idxs, {added_carbon -> skeleton-attachment C})
    or ``None`` if any added carbon is not a carbon bonded to exactly one acyclic
    non-added skeleton carbon (the cases this namer does not own)."""
    added = sorted({m[0] for m in pg_matches if m})
    added_set = set(added)
    attach = {}
    for ac in added:
        atom = mol.GetAtomWithIdx(ac)
        if atom.GetSymbol() != "C":
            return None
        carbon_nbrs = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() not in added_set
        ]
        if len(carbon_nbrs) != 1:
            return None  # added C bonded to 0 or >1 skeleton carbons -> not our case
        sk = carbon_nbrs[0]
        if mol.GetAtomWithIdx(sk).IsInRing():
            return None  # ring-attached -> ring suffix path owns it
        attach[ac] = sk
    return added, attach


def _frag_on_benzene_smiles(mol, frag_atoms, attach_idx) -> Optional[str]:
    """Build the SMILES of the substituent fragment attached to a benzene ring, in
    RDKit (no OPSIN). Used to CONSTITUTION-check a substituent name gate-independently."""
    from rdkit import Chem

    rw = Chem.RWMol()
    idxmap = {}
    for a in sorted(frag_atoms):
        src = mol.GetAtomWithIdx(a)
        na = Chem.Atom(src.GetAtomicNum())
        na.SetFormalCharge(src.GetFormalCharge())
        idxmap[a] = rw.AddAtom(na)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idxmap and j in idxmap:
            rw.AddBond(idxmap[i], idxmap[j], b.GetBondType())
    ring = [rw.AddAtom(Chem.Atom(6)) for _ in range(6)]
    for k in range(6):
        rw.AddBond(ring[k], ring[(k + 1) % 6], Chem.BondType.AROMATIC)
        rw.GetAtomWithIdx(ring[k]).SetIsAromatic(True)
    rw.AddBond(ring[0], idxmap[attach_idx], Chem.BondType.SINGLE)
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _substituent_constitution_ok(mol, frag_atoms, attach_idx, sub_name: str) -> bool:
    """Gate-INDEPENDENT CONSTITUTION re-anchor for a substituent prefix name.

    `name_substituent` has a SYMBOLS-ONLY multi-atom fallback that mis-names
    constitutional isomers (a nitrite ``-O-N=O`` came back ``nitro`` — same {N,O,O}
    heavy-atom count, different connectivity), so a None/sentinel check (and even a
    heavy-COUNT check) is insufficient. OPSIN-parse ``<sub_name>benzene`` and require
    its InChIKey skeleton to equal that of the REAL fragment-on-benzene; fail CLOSED
    on any parse failure / missing jar / mismatch. Fixes the fable BLOCKER 1 (a
    nitrite shipping as a nitro compound gate-off) without depending on SELF-01."""
    from ..namer import _validity_gate_name_to_smiles, _self_consistency_skeleton

    probe = _validity_gate_name_to_smiles(f"{sub_name}benzene")
    if not probe:
        return False
    actual = _frag_on_benzene_smiles(mol, frag_atoms, attach_idx)
    if actual is None:
        return False
    sk_named = _self_consistency_skeleton(probe)
    sk_actual = _self_consistency_skeleton(actual)
    return sk_named is not None and sk_named == sk_actual


def _whole_name_stereo_ok(mol, name: str) -> bool:
    """Gate-INDEPENDENT stereo-aware re-anchor for the UNSATURATED added-carbon
    branch (aconitic family). The ene locant and the E/Z descriptor are spelled by
    this namer, so a spelling slip could ship a wrong constitution/geometry gate-off
    (T4 producers must re-anchor or fail closed -- ed52fa98 / 8afa533c). OPSIN-parse
    the WHOLE name and require its InChIKey (constitution AND stereo, full key) to
    equal the input's; fail CLOSED on no-jar / parse-fail / mismatch."""
    from rdkit import Chem
    from ..namer import _validity_gate_name_to_smiles

    smi = _validity_gate_name_to_smiles(name)
    if not smi:
        return False
    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return False
    try:
        return Chem.MolToInchiKey(parsed) == Chem.MolToInchiKey(mol)
    except Exception:
        return False


def name_added_carbon_parent(features: Any, style: str = "pin") -> Optional[str]:
    """Name an acyclic >=3-group added-carbon multi-suffix molecule, or ``None``
    to fall through to the legacy path when the structure is outside the clean
    acyclic-parent class this namer owns."""
    mol = features.mol
    fg_name = features.principal_group
    pg_matches = features.principal_group_atoms
    carbo = _added_carbon_suffix(fg_name)
    if carbo is None or not pg_matches:
        return None

    parsed = _added_carbons_and_attachments(mol, pg_matches)
    if parsed is None:
        return None
    added, attach = parsed
    n_groups = len(added)
    if n_groups < 3:
        return None
    added_set = set(added)

    # Parent hydride = the carbon skeleton with the added carbons removed.
    chain = find_longest_carbon_chain(mol, exclude_atoms=added_set)
    if not chain:
        return None

    # All non-added carbons must lie ON this single chain (no branches). Every
    # parent carbon's heavy neighbour must be a parent/added carbon OR a nameable
    # exocyclic SUBSTITUENT (v30: citric-acid family -- a 2-hydroxy on the core;
    # previously ANY extra neighbour returned None -> the molecule fell to a wrong
    # pentanedioic-chain candidate). Collect each substituent fragment and name it
    # via the recursive substituent namer; fail closed on anything un-nameable so a
    # wrong/atom-dropped name is never emitted.
    from rdkit import Chem

    skeleton_carbons = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "C" and a.GetIdx() not in added_set
    }
    chain_set = set(chain)
    if skeleton_carbons != chain_set:
        return None
    # substituent = (chain_atom_idx, attach_neighbour_idx, frozenset(frag_atoms))
    substituents: List[tuple] = []
    for c in chain:
        catom = mol.GetAtomWithIdx(c)
        if catom.IsInRing():
            return None
        for nbr in catom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in chain_set or ni in added_set or nbr.GetSymbol() == "H":
                continue
            # BFS the substituent fragment from ni, never crossing back into the
            # parent chain or the added (suffix) carbons.
            frag = set()
            stack = [ni]
            while stack:
                a = stack.pop()
                if a in frag or a in chain_set or a in added_set:
                    continue
                frag.add(a)
                for nn in mol.GetAtomWithIdx(a).GetNeighbors():
                    j = nn.GetIdx()
                    if j not in frag and j not in chain_set and j not in added_set:
                        stack.append(j)
            substituents.append((c, ni, frozenset(frag)))

    # Parent-chain unsaturation. SINGLE bonds are the saturated (propane) core;
    # DOUBLE bonds make the aconitic family (prop-1-ene-1,2,3-tricarboxylic acid).
    # Triple / aromatic parent bonds are out of scope -> fail closed.
    core_double_bonds = []  # (chain_atom_a, chain_atom_b, bond)
    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if bond is None:
            return None
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            continue
        if bt == Chem.BondType.DOUBLE:
            core_double_bonds.append((chain[i], chain[i + 1], bond))
        else:
            return None  # triple / aromatic parent bond -> out of scope

    length = len(chain)
    stem = get_chain_prefix(length)
    multiplier = SIMPLE_MULTIPLIERS.get(n_groups)
    if multiplier is None:
        return None

    from ..assembly.naming_utils import (
        alpha_sort_key, is_complex_substituent, get_multiplier_prefix,
    )

    # Name every substituent (fail closed if any is un-nameable OR mis-named) BEFORE
    # choosing the numbering, so a molecule this namer cannot fully+correctly express
    # is declined here rather than emitting a partial/wrong name.
    named_subs: List[tuple] = []  # (chain_atom_idx, bare_prefix_name)
    if substituents:
        from ..assembly.substituent_enumerator import name_substituent
        for chain_atom, ni, frag in substituents:
            sub_name = name_substituent(mol, sorted(frag), ni)
            if not sub_name or sub_name == "substituent":
                return None  # un-nameable substituent -> fail closed (0-wrong)
            # v30 fable BLOCKER 1: name_substituent's symbols-only fallback mis-names
            # constitutional isomers (nitrite -O-N=O -> 'nitro'); a gate-independent
            # CONSTITUTION re-anchor rejects that before it can ship gate-off.
            if not _substituent_constitution_ok(mol, frag, ni, sub_name):
                return None
            named_subs.append((chain_atom, sub_name))

    # P-14.4 NUMBERING (L3219): lowest locants to (c) the added-carbon (suffix)
    # attachments, then (f) all substituents as a set, then (g) the substituent cited
    # FIRST in alphanumerical order (P-14.4(g), BB: "1-methyl-4-nitronaphthalene, not
    # 4-methyl-1-nitro..."). Without (g) the PIN depended on SMILES atom order.
    # (Not P-31.1.4 -- that section is von Baeyer parent hydrides, L16619.)
    pos = {atom_idx: i for i, atom_idx in enumerate(chain)}

    def loc(atom_idx: int, reverse: bool) -> int:
        p = pos[atom_idx]
        return (length - 1 - p) + 1 if reverse else p + 1

    subs_alpha = sorted(named_subs, key=lambda t: alpha_sort_key(t[1]))

    # Stereo — computed BEFORE numbering so P-14.4(j) can break a locant tie (fable
    # review of a4240802). The core is saturated (unsaturation rejected above), so this
    # is R/S CHAIN stereocentres. Fail CLOSED if any DEFINED stereocentre is off the
    # parent chain (would live inside a substituent), is unassignable, is pseudo-
    # asymmetric r/s or axial M/P (OPSIN-unparseable / out of scope), or any stereo
    # double bond exists — a flat name would drop or mis-spell it. Uses the vendored-
    # `centres` CIP path. (Fable RISK 3 — a stale parse-time legacy _CIPCode surviving
    # for an atom centres declined — is LATENT (no witness in the >=3-COOH acyclic class,
    # invariant 10); NOT hardened here because clearing _CIPCode breaks assign's
    # repopulation. Tracked as a follow-up.)
    from ..perception.stereo import assign_stereochemistry
    assign_stereochemistry(mol)
    # A stereogenic double bond ON the parent chain is EXPRESSED as (locant)E/Z; any
    # OTHER stereo bond (inside a substituent) cannot be mapped to a chain locant ->
    # fail closed. (Was: reject ALL stereo bonds; now scoped to the core.)
    core_bond_ids = {bond.GetIdx() for _, _, bond in core_double_bonds}
    for b in mol.GetBonds():
        if (b.GetStereo() != Chem.BondStereo.STEREONONE
                and b.GetIdx() not in core_bond_ids):
            return None
    # E/Z of each core double bond -- only when a geometry is DEFINED; an undefined
    # core double bond names flat (matching an input that left it unspecified).
    db_cip = {}  # (chain_atom_a, chain_atom_b) -> 'E' | 'Z'
    for a_idx, b_idx, bond in core_double_bonds:
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if not bond.HasProp("_CIPCode"):
            return None
        ez = bond.GetProp("_CIPCode")
        if ez not in ("E", "Z"):
            return None
        db_cip[(a_idx, b_idx)] = ez

    chain_cip = {}  # chain_atom_idx -> 'R' | 'S'
    for a in mol.GetAtoms():
        if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        aidx = a.GetIdx()
        if aidx not in pos or not a.HasProp("_CIPCode"):
            return None
        cip = a.GetProp("_CIPCode")
        if cip not in ("R", "S"):
            return None  # pseudo-asymmetric (r/s) / axial (M/P) -> fail closed
        chain_cip[aidx] = cip

    _CIP_RANK = {"R": 0, "S": 1}  # P-14.4(j): R preferred (lower) over S
    _EZ_RANK = {"Z": 0, "E": 1}   # P-14.4(j): Z preferred (lower) over E

    def _db_loc(a_idx, b_idx, reverse):
        return min(loc(a_idx, reverse), loc(b_idx, reverse))

    def ene_locants(reverse):
        return sorted(_db_loc(a, b, reverse) for a, b, _ in core_double_bonds)

    def key_for(reverse: bool):
        suf = sorted(loc(attach[ac], reverse) for ac in added)
        ene = ene_locants(reverse)          # P-14.4(e)(i): ene after suffix P-14.4(c)
        sub = sorted(loc(ca, reverse) for ca, _ in named_subs)
        alpha = [loc(ca, reverse) for ca, _ in subs_alpha]  # P-14.4(g)
        # P-14.4(j): lowest locants to the preferred stereodescriptor; R/S and E/Z
        # descriptors ordered together by locant.
        st_items = [(loc(x, reverse), _CIP_RANK[chain_cip[x]]) for x in chain_cip]
        st_items += [(_db_loc(a, b, reverse), _EZ_RANK[db_cip[(a, b)]])
                     for (a, b) in db_cip]
        stereo = [r for _, r in sorted(st_items)]
        return (suf, ene, sub, alpha, stereo)

    reverse = key_for(True) < key_for(False)
    suffix_locants = sorted(loc(attach[ac], reverse) for ac in added)

    # Merge R/S and E/Z stereodescriptors into one locant-ordered leading paren.
    # BB: fumaric acid = "(2E)-but-2-enedioic acid" (PIN) -> the descriptor carries
    # its locant even for a single double bond.
    stereo_prefix = ""
    st_bits = [(loc(x, reverse), chain_cip[x]) for x in chain_cip]
    st_bits += [(_db_loc(a, b, reverse), db_cip[(a, b)]) for (a, b) in db_cip]
    if st_bits:
        st_bits.sort()
        stereo_prefix = "(" + ",".join(f"{lc}{d}" for lc, d in st_bits) + ")-"

    # Assemble the substituent-prefix string (P-16.3.3 enclosure, P-14.5.2 alpha
    # order, P-16.3.4 multipliers). P-14.3.4.2(a): the locant '1' is omitted for a
    # substituted MONONUCLEAR (methane) parent hydride.
    prefix_str = ""
    if named_subs:
        from collections import defaultdict
        omit_locants = (length == 1)
        by_name: dict = defaultdict(list)
        for ca, nm in named_subs:
            by_name[nm].append(loc(ca, reverse))
        parts = []
        for nm in sorted(by_name, key=alpha_sort_key):
            locs = sorted(by_name[nm])
            complex_ = is_complex_substituent(nm)
            disp = f"({nm})" if complex_ and not (nm.startswith("(") and nm.endswith(")")) else nm
            mult = get_multiplier_prefix(len(locs), complex_)
            if omit_locants:
                parts.append(f"{mult}{disp}")
            else:
                locstr = ",".join(str(x) for x in locs)
                parts.append(f"{locstr}-{mult}{disp}")
        # P-16.3.4: hyphen-join the ordered prefix fragments.
        prefix_str = "-".join(parts) if len(parts) > 1 else parts[0]
        # the parent stem starts with a letter, so no hyphen needed after prefix_str.

    # Parent hydride string, built AFTER numbering so ene locants use the chosen
    # direction. Monoene inserts "-<loc>-ene" (prop-1-ene); polyene inserts the
    # euphonic 'a' before the consonant-initial multiplied ending (buta-1,3-diene).
    if core_double_bonds:
        elocs = ene_locants(reverse)
        emult = {1: "", 2: "di", 3: "tri", 4: "tetra"}.get(len(elocs))
        if emult is None:
            return None
        eloc_str = ",".join(str(x) for x in elocs)
        if length == 2:
            # Dinuclear chain: the double bond can only be 1-2, so its locant is
            # structurally redundant (deny-by-default P-14.3.3 omits an unnecessary
            # locant). BB writes ethene-1,1,2-triyl / ethene-1,2-diyl even when
            # substituted, and `eth-1-ene` 0 times -> "ethene" (never "eth-1-ene").
            # A stereogenic dinuclear C=C keeps its DESCRIPTOR locant, e.g. BB
            # `(1E)-...ethene-1,2-diyl`, which stereo_prefix already carries.
            parent_core = f"{stem}ene"
        elif len(elocs) == 1:
            parent_core = f"{stem}-{eloc_str}-ene"
        else:
            parent_core = f"{stem}a-{eloc_str}-{emult}ene"
    else:
        parent_core = f"{stem}ane"

    if length == 1:
        name = f"{stereo_prefix}{prefix_str}{parent_core}{multiplier}{carbo}"
    else:
        locant_str = ",".join(str(x) for x in suffix_locants)
        name = f"{stereo_prefix}{prefix_str}{parent_core}-{locant_str}-{multiplier}{carbo}"

    # Gate-INDEPENDENT re-anchor for the NEW unsaturated branch only (saturated path
    # stays byte-identical). Fails closed if OPSIN cannot confirm the whole name
    # denotes the exact input (constitution + stereo).
    if core_double_bonds and not _whole_name_stereo_ok(mol, name):
        return None
    return name


__all__ = ["requires_added_carbon_suffix", "name_added_carbon_parent"]
