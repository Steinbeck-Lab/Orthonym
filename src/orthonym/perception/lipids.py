"""Lipid backbone detection for the P-107 backbone-aware assembler (WSC-01).

`detect_lipid_backbone(mol)` is a PURE, hard-gated structural deriver: it recognizes
the three lipid backbone families and classifies each backbone position, or returns
``None`` on any dirty/unrecognized decoration so the molecule defers to the general
pipeline (fail-safe → zero non-lipid regression). It is the acyclic
generalization of the Phase-176 ``recognize_sugar_skeleton`` ring deriver.

The detector does NOT name anything — it produces a structured ``BackboneMatch`` that
the Form-B assembler (``rules/lipids.py``) consumes. The only mutation is the
idempotent CIP-label assignment (the accepted sugar-deriver seam).

Families (Blue Book P-107):
  - "glyceride" — propane-1,2,3-triyl core, O's acylated / phospho / free-OH (P-107.2)
  - "phospholipid" — glyceride where one primary O is a phosphate diester to a head group (P-107.3)
  - "sphingolipid" — long-chain 2-amino-1,3-diol (sphinganine/sphingosine); N-acyl = ceramide (P-107.4.3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

logger = logging.getLogger(__name__)

_CIP_ASSIGNED_PROP = "_orthonym_cip_assigned"

# Anchor: three contiguous sp3 carbons, each bearing exactly one O — the
# propane-1,2,3-triyl-O core. (Terminal-internal-terminal CH2-CH-CH2.)
_GLYCEROL_ANCHOR = Chem.MolFromSmarts("[CH2X4][CHX4][CH2X4]")
# Sphingoid: HO-CH2(C1)-CH(N)(C2)-CH(O)(C3)-; C1-O may be OH or O-glycosyl.
_SPHINGOID_ANCHOR = Chem.MolFromSmarts("[OX2][CH2X4][CHX4]([NX3])[CHX4][OX2H1]")


@dataclass(frozen=True)
class BackboneMatch:
    """Structured result of lipid backbone detection (consumed by rules/lipids.py)."""

    family: str                       # "glyceride" | "phospholipid" | "sphingolipid"
    core_atoms: tuple                 # backbone carbon atom indices (C1,C2,C3 order)
    position_locants: dict            # {atom_idx: locant_int}
    sites: dict                       # locant -> site tuple (see module docstring)
    backbone_atom_to_locant: dict     # for inject_stereo_from_locant_map
    cip: dict = field(default_factory=dict)   # {atom_idx: "R"|"S"|"-"}
    extra: dict = field(default_factory=dict)  # family-specific (e.g. chain_len, db_locant)


# --------------------------------------------------------------------------- #
# CIP (idempotent — mirrors sugar_names._ensure_cip / perception.stereo)
# --------------------------------------------------------------------------- #
def _ensure_cip(mol) -> None:
    if mol.HasProp(_CIP_ASSIGNED_PROP):
        return
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except (ValueError, RuntimeError, KeyError) as exc:  # pragma: no cover
        logger.debug("CIP assignment failed in lipid deriver: %s", exc)
    mol.SetProp(_CIP_ASSIGNED_PROP, "1")


def _cip(atom) -> str:
    return atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "-"


# --------------------------------------------------------------------------- #
# Small structural helpers
# --------------------------------------------------------------------------- #
def _heavy_neighbors(atom):
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]


def _o_neighbors(atom):
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8]


def _classify_oxygen_site(mol, c_idx, o_idx):
    """Classify the substituent reached through backbone carbon `c_idx`'s oxygen `o_idx`.

    Returns a site tuple or None (None = dirty decoration → caller hard-gates).
    """
    o = mol.GetAtomWithIdx(o_idx)
    o_heavy = [n for n in o.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() != c_idx]

    # Free hydroxyl: O with only the backbone carbon as heavy neighbor.
    if not o_heavy:
        if o.GetFormalCharge() == 0:
            return ("free_oh", o_idx)
        return None  # charged bare O on a glyceride (not a clean OH) → defer

    other = o_heavy[0]
    z = other.GetAtomicNum()

    # Phosphate ester: O-P
    if z == 15:
        return ("phospho", other.GetIdx(), o_idx)

    # Carbon neighbor: acyl ester (O-C=O), glycosyl (O-anomeric-ring-C), or ether (defer)
    if z == 6:
        # acyl: the carbon is a carbonyl C (has a double-bonded O)
        for nb in other.GetNeighbors():
            if nb.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(other.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                return ("acyl", other.GetIdx(), o_idx)
        # glycosyl: the carbon is an anomeric ring carbon (in a ring, bonded to a ring O)
        if other.IsInRing():
            sugar = _recognize_attached_sugar(mol, o_idx, other.GetIdx())
            if sugar is not None:
                return ("glycosyl", o_idx, sugar)
            return None  # ring carbon we can't resolve as a clean sugar → defer
        # plain ether (O-alkyl) or vinyl-ether → defer (ether/plasmalogen)
        return None

    return None


def _recognize_attached_sugar(mol, o_idx, anomeric_idx):
    """Run recognize_sugar_skeleton on the sugar fragment reached through `o_idx`.

    Splits the glycosidic O-C(anomeric) bond and CAPS the anomeric carbon with a
    fresh -OH so the isolated fragment is a free sugar (anomeric -OH present, the
    form recognize_sugar_skeleton expects). Returns (anomer, config, base) or None.
    """
    from orthonym.data.sugar_names import recognize_sugar_skeleton
    bond = mol.GetBondBetweenAtoms(o_idx, anomeric_idx)
    if bond is None:
        return None
    rw = Chem.RWMol(mol)
    rw.RemoveBond(o_idx, anomeric_idx)
    new_o = rw.AddAtom(Chem.Atom(8))            # restore the anomeric -OH
    rw.AddBond(anomeric_idx, new_o, Chem.BondType.SINGLE)
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return None
    for frag in Chem.GetMolFrags(m2):
        if anomeric_idx in frag:
            try:
                sub = Chem.MolFragmentToSmiles(
                    m2, atomsToUse=list(frag), canonical=True, isomericSmiles=True)
                piece = Chem.MolFromSmiles(sub)
            except Exception:
                return None
            if piece is None:
                return None
            return recognize_sugar_skeleton(piece)
    return None


# --------------------------------------------------------------------------- #
# Head-group classification for phospholipids (P-107.3)
# --------------------------------------------------------------------------- #
_HEADGROUP_SMARTS = [
    # (descriptor, SMARTS on the head-group fragment rooted at the phospho-O carbon)
    ("choline", Chem.MolFromSmarts("[OX2][CH2][CH2][N+]([CH3])([CH3])[CH3]")),
    ("ethanolamine", Chem.MolFromSmarts("[OX2][CH2][CH2][NX3;H2,H3+]")),
    ("serine", Chem.MolFromSmarts("[OX2][CH2][CHX4]([NX3])C(=O)[OX1,OX2H1]")),
    ("inositol", Chem.MolFromSmarts("[OX2][CH1]1[CH1][CH1][CH1][CH1][CH1]1")),
]


def _classify_head_group(mol, p_idx, backbone_o_idx):
    """Classify the phosphate head group; returns (descriptor, head_o_idx) or None.

    The phosphate P has the backbone ester O plus 1-2 other O's. The head group is
    the non-backbone ester O that leads to a recognized moiety; remaining O's are
    OH/O- (phosphatidic acid keeps them free).
    """
    p = mol.GetAtomWithIdx(p_idx)
    ester_os = []
    for nb in p.GetNeighbors():
        if nb.GetAtomicNum() != 8 or nb.GetIdx() == backbone_o_idx:
            continue
        if mol.GetBondBetweenAtoms(p_idx, nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
            continue  # the P=O
        # esterified (O bonded to a carbon) vs free OH/O-
        if any(n.GetAtomicNum() == 6 for n in nb.GetNeighbors() if n.GetIdx() != p_idx):
            ester_os.append(nb.GetIdx())

    if not ester_os:
        return ("phosphate", None)  # phosphatidic acid (free phosphate)

    for head_o in ester_os:
        for desc, patt in _HEADGROUP_SMARTS:
            if patt is None:
                continue
            for match in mol.GetSubstructMatches(patt):
                if match and match[0] == head_o:
                    return (desc, head_o)
        # glycerol head group: O-CH2-CH(OH)-CH2-OH
        if _is_glycerol_headgroup(mol, head_o):
            return ("glycerol", head_o)
    # an esterified head we don't recognize → defer
    return None


def _is_glycerol_headgroup(mol, o_idx):
    patt = Chem.MolFromSmarts("[OX2][CH2X4][CHX4]([OX2H1])[CH2X4][OX2H1]")
    if patt is None:
        return False
    return any(m and m[0] == o_idx for m in mol.GetSubstructMatches(patt))


# --------------------------------------------------------------------------- #
# Glycerol / glycero-phospho family
# --------------------------------------------------------------------------- #
def _detect_glyceride(mol) -> Optional[BackboneMatch]:
    for match in mol.GetSubstructMatches(_GLYCEROL_ANCHOR):
        c1, c2, c3 = match
        atoms = [mol.GetAtomWithIdx(i) for i in match]
        # propane shape: terminals are CH2 with exactly {central, O} heavy; central CH with {c1,c3,O}
        if any(a.IsInRing() for a in atoms):
            continue
        o_of = {}
        ok = True
        for ci, c in zip(match, atoms):
            os = _o_neighbors(c)
            if len(os) != 1:
                ok = False
                break
            o_of[ci] = os[0].GetIdx()
        if not ok:
            continue
        # terminal carbons must have NO carbon neighbor besides the central c2 (rejects longer polyols)
        def heavy_c(c):
            return [n.GetIdx() for n in c.GetNeighbors() if n.GetAtomicNum() == 6]
        if heavy_c(atoms[0]) != [c2] or heavy_c(atoms[2]) != [c2]:
            continue
        if sorted(heavy_c(atoms[1])) != sorted([c1, c3]):
            continue

        # classify each O site
        sites = {}
        locant_of = {c1: 1, c2: 2, c3: 3}
        family = "glyceride"
        bad = False
        for ci in (c1, c2, c3):
            site = _classify_oxygen_site(mol, ci, o_of[ci])
            if site is None:
                bad = True
                break
            if site[0] == "phospho":
                hg = _classify_head_group(mol, site[1], o_of[ci])
                if hg is None:
                    bad = True
                    break
                family = "phospholipid"
                sites[locant_of[ci]] = ("phospho", site[1], o_of[ci], hg[0], hg[1])
            else:
                sites[locant_of[ci]] = site if site[0] != "acyl" else ("acyl", site[1], o_of[ci])
        if bad:
            continue

        # at least one acyl OR phospho (a bare propane-1,2,3-triol is just glycerol, not a lipid)
        kinds = {v[0] for v in sites.values()}
        if "acyl" not in kinds and "phospho" not in kinds:
            continue

        _ensure_cip(mol)
        cip = {c2: _cip(mol.GetAtomWithIdx(c2))}
        return BackboneMatch(
            family=family,
            core_atoms=(c1, c2, c3),
            position_locants=locant_of,
            sites=sites,
            backbone_atom_to_locant={c1: 1, c2: 2, c3: 3},
            cip=cip,
        )
    return None


# --------------------------------------------------------------------------- #
# Sphingoid family (P-107.4.3)
# --------------------------------------------------------------------------- #
def _detect_sphingoid(mol) -> Optional[BackboneMatch]:
    for match in mol.GetSubstructMatches(_SPHINGOID_ANCHOR):
        o1, c1, c2, n, c3, o3 = match
        a_c1, a_c2, a_c3 = (mol.GetAtomWithIdx(c1), mol.GetAtomWithIdx(c2),
                            mol.GetAtomWithIdx(c3))
        if any(a.IsInRing() for a in (a_c1, a_c2, a_c3)):
            continue
        # C1 must be a primary CH2 (terminal: heavy neighbors = {O1, C2})
        c1_heavy = sorted(nb.GetIdx() for nb in a_c1.GetNeighbors() if nb.GetAtomicNum() > 1)
        if c1_heavy != sorted([o1, c2]):
            continue
        # C3 must continue into a long aliphatic chain (the sphingoid tail) → has a C neighbor besides C2
        c3_carbons = [nb.GetIdx() for nb in a_c3.GetNeighbors()
                      if nb.GetAtomicNum() == 6 and nb.GetIdx() != c2]
        if not c3_carbons:
            continue

        # C1 oxygen: free OH or glycosyl
        sites = {}
        o1_atom = mol.GetAtomWithIdx(o1)
        o1_heavy = [nb for nb in o1_atom.GetNeighbors()
                    if nb.GetAtomicNum() > 1 and nb.GetIdx() != c1]
        if not o1_heavy:
            sites[1] = ("free_oh", o1)
        else:
            oc = o1_heavy[0]
            if oc.GetAtomicNum() == 6 and oc.IsInRing():
                sugar = _recognize_attached_sugar(mol, o1, oc.GetIdx())
                if sugar is None:
                    continue
                sites[1] = ("glycosyl", o1, sugar)
            else:
                continue  # C1-O-alkyl ether on a sphingoid → defer

        # C3 oxygen: must be free OH (neutral sphingoid)
        o3_atom = mol.GetAtomWithIdx(o3)
        if [nb for nb in o3_atom.GetNeighbors() if nb.GetAtomicNum() > 1 and nb.GetIdx() != c3]:
            continue
        sites[3] = ("free_oh", o3)

        # C2 nitrogen: N-acyl (ceramide) or free NH2 (bare sphingoid)
        n_atom = mol.GetAtomWithIdx(n)
        acyl_c = None
        for nb in n_atom.GetNeighbors():
            if nb.GetAtomicNum() == 6 and nb.GetIdx() not in (c2,):
                # is it a carbonyl carbon (amide)?
                if any(o.GetAtomicNum() == 8 and
                       mol.GetBondBetweenAtoms(nb.GetIdx(), o.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                       for o in nb.GetNeighbors()):
                    acyl_c = nb.GetIdx()
        if acyl_c is not None:
            sites[2] = ("n_acyl", acyl_c, n)
        else:
            # bare sphingoid (free amine) — defer: the systematic 2-aminoalkane-1,3-diol
            # already names; claiming it risks regression. Documented in SUMMARY.
            return None

        # Walk the linear sphingoid chain C1->Cn (C1 = the CH2-O end) for the
        # substituent stem length + double-bond locants. Branched sphingoid → defer.
        chain = [c1, c2, c3]
        prev, cur = c2, c3
        while True:
            nxts = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
                    if nb.GetAtomicNum() == 6 and nb.GetIdx() != prev and nb.GetIdx() not in chain]
            if len(nxts) != 1:
                if len(nxts) > 1:
                    return None  # branched sphingoid → honest-gate
                break
            prev, cur = cur, nxts[0]
            chain.append(cur)
        chain_pos = {idx: i + 1 for i, idx in enumerate(chain)}  # C1=1...
        double_bonds = []
        for i in range(len(chain) - 1):
            b = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if b is not None and b.GetBondType() == Chem.BondType.DOUBLE:
                double_bonds.append(i + 1)  # locant = lower carbon position

        _ensure_cip(mol)
        cip = {c2: _cip(a_c2), c3: _cip(a_c3)}
        return BackboneMatch(
            family="sphingolipid",
            core_atoms=(c1, c2, c3),
            position_locants={c1: 1, c2: 2, c3: 3},
            sites=sites,
            backbone_atom_to_locant={c1: 1, c2: 2, c3: 3},
            cip=cip,
            extra={"amino_n": n, "chain": chain, "chain_pos": chain_pos,
                   "chain_length": len(chain), "double_bonds": double_bonds},
        )
    return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def detect_lipid_backbone(mol) -> Optional[BackboneMatch]:
    """Detect a clean lipid backbone, or return None (hard gate / fail-safe)."""
    if mol is None:
        return None
    # sphingoid first (its 1,3-diol-2-amino core is more specific than the glycerol triol)
    sph = _detect_sphingoid(mol)
    if sph is not None:
        return sph
    return _detect_glyceride(mol)
