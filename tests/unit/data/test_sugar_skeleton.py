"""
D-13 catalog-reproduction HARD gate for the structure-derived sugar recognizer.

Phase 176 Plan 01 (WSD-08). These tests are the fail-stop that keeps
``recognize_sugar_skeleton`` honest: the deriver MUST reproduce the existing
retained-sugar catalog's ``(anomer, config, base)`` tuple for every
structurally-clean stereo entry, and MUST fail closed (return ``None``, never a
wrong tuple) on every modified / out-of-scope ring sugar (deoxy / N-acetyl /
uronic / amino / C-modified).

Two notions of "clean" exist in the catalog and they are NOT the same set:

* **Keyword-clean** (the plan's base-name filter): 44 stereo pyranose/furanose
  entries whose *base name* contains no deoxy/amino/acetamido/uron token. This
  count is asserted as a catalog-drift tripwire (``KEYWORD_CLEAN == 44``), but
  it still includes the 4 deoxy-L sugars rhamnopyranose/fucopyranose, whose
  *names* carry no "deoxy" token even though they are structurally deoxy
  (ring-CH3, not ring-CH2OH).

* **Structurally-clean** (the deriver's actual D-06 contract): the clean
  hexose/pentose OH/H/CH2OH fingerprint — 40 stereo pyranose/furanose entries.
  rhamnopyranose/fucopyranose are EXCLUDED here (they are deoxy → the deriver
  returns ``None`` per D-06 / RESEARCH §"Verified deriver fingerprint").

The catalog-reproduction gate (D-13) therefore reproduces the 40 structurally
clean entries; the None gate covers the remaining stereo ring sugars (the 4
deoxy + 4 N-acetyl + 2 uronic = 10 modified ring sugars), proving the deriver
never emits a wrong tuple on out-of-scope input.
"""

import pytest
from rdkit import Chem

from orthonym.data.sugar_names import (
    ALL_SUGAR_NAMES,
    lookup_sugar,
    recognize_sugar_skeleton,
    sugar_to_glycoside_class_name,
)


# ---------------------------------------------------------------------------
# Clean-set construction (deterministic; derived from the catalog, not hardcoded)
# ---------------------------------------------------------------------------

# Base-name tokens that mark a structurally modified sugar. ``uron`` (not
# ``uronic``) is required so that ``glucuronopyranose`` is caught.
_MODIFIED_BASE_TOKENS = ("deoxy", "amino", "acetamido", "uron", "acetylamino")


def _is_ring_sugar_entry(smiles, base_name):
    """True iff this catalog entry is a stereo-keyed pyranose/furanose ring."""
    if "@" not in smiles:
        return False
    return base_name.endswith("pyranose") or base_name.endswith("furanose")


def _is_keyword_clean(base_name):
    """The plan's base-name keyword filter (catalog-drift tripwire only)."""
    bl = base_name.lower()
    return not any(tok in bl for tok in _MODIFIED_BASE_TOKENS)


def _is_structurally_clean(smiles):
    """The deriver's D-06 contract: clean hexose/pentose OH/H/CH2OH fingerprint.

    A single 5- or 6-membered ring with exactly one ring oxygen, every ring
    carbon bearing only OH (or one CH2OH exocyclic carbon), no nitrogen, no
    deoxy ring-CH3, and no carbonyl/carboxyl anywhere. This is what
    ``recognize_sugar_skeleton`` must reproduce; everything else → ``None``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    # No nitrogen anywhere (amino / N-acetyl sugars).
    if any(a.GetSymbol() == "N" for a in mol.GetAtoms()):
        return False
    # No double bonds anywhere (carbonyl / carboxyl uronic acids, glycals).
    if any(b.GetBondTypeAsDouble() == 2.0 for b in mol.GetBonds()):
        return False
    ri = mol.GetRingInfo()
    if ri.NumRings() != 1:
        return False
    ring = set(ri.AtomRings()[0])
    if len(ring) not in (5, 6):
        return False
    ring_oxygens = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
    if len(ring_oxygens) != 1:
        return False
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C":
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in ring:
                continue
            sym = nbr.GetSymbol()
            if sym == "O":
                continue  # exocyclic hydroxyl / anomeric-O
            if sym == "C":
                # exocyclic carbon must be a CH2OH (carries an O neighbor);
                # a bare CH3 marks a 6-deoxy sugar (rhamnose/fucose).
                if not any(nn.GetSymbol() == "O" for nn in nbr.GetNeighbors()):
                    return False
            else:
                return False
    return True


def _ring_sugar_entries():
    return [
        (smi, val)
        for smi, val in ALL_SUGAR_NAMES.items()
        if _is_ring_sugar_entry(smi, val[2])
    ]


_RING_SUGARS = _ring_sugar_entries()
KEYWORD_CLEAN = [(s, v) for s, v in _RING_SUGARS if _is_keyword_clean(v[2])]
STRUCTURAL_CLEAN = [(s, v) for s, v in _RING_SUGARS if _is_structurally_clean(s)]
MODIFIED_RING = [(s, v) for s, v in _RING_SUGARS if not _is_structurally_clean(s)]


@pytest.mark.unit
class TestSugarSkeletonCatalogReproduction:
    """D-13 HARD gate: the deriver reproduces the clean catalog, fails closed."""

    def test_clean_set_scope_self_validates(self):
        """The clean-set scope is exactly what the catalog pins (catalog-drift trip).

        v23 CARB-02 grew the catalog with the complete systematic monosaccharide
        stereo family (SYSTEMATIC_MONOSACCHARIDE_NAMES, +176 OPSIN-verified aldo/
        keto tetro/pento/hexo D/L pyranose/furanose alpha/beta/unspec entries),
        so both scopes grew by 176: keyword-clean 44 -> 220, structural-clean
        40 -> 216.  The 4-entry difference (the deoxy-L rhamnose/fucose entries,
        keyword-clean names but structurally deoxy ring-CH3) is unchanged.
        """
        # Keyword-clean is the base-name filter: 220 (includes the 4 deoxy-L
        # rhamnose/fucose entries whose names carry no 'deoxy' token).
        clean = KEYWORD_CLEAN  # noqa: F841 - named for the acceptance grep
        assert len(clean) == 220, (
            f"keyword-clean catalog scope drifted: got {len(clean)} (expected 220)"
        )
        # Structural-clean is the deriver's D-06 contract: 216
        # (rhamnose/fucose excluded — they are deoxy ring-CH3).
        assert len(STRUCTURAL_CLEAN) == 216, (
            f"structural-clean scope drifted: got {len(STRUCTURAL_CLEAN)} "
            "(expected 216)"
        )
        # The 4 deoxy entries are the difference between the two notions.
        assert len(KEYWORD_CLEAN) - len(STRUCTURAL_CLEAN) == 4

    def test_catalog_reproduction(self):
        """Deriver reproduces (anomer, config, base) for every clean entry (D-13)."""
        for smiles, expected in STRUCTURAL_CLEAN:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"unparseable catalog SMILES: {smiles}"
            result = recognize_sugar_skeleton(mol, anomeric_idx=None)
            assert result == expected, (
                f"deriver diverged from catalog for {smiles}: "
                f"got {result!r}, expected {expected!r}"
            )
            # And it must equal the catalog fast-path (apples-to-apples shape).
            assert result == lookup_sugar(Chem.MolToSmiles(mol))

    def test_modified_entries_return_none(self):
        """Deriver fails closed on every modified/out-of-scope ring sugar (D-06)."""
        assert MODIFIED_RING, "expected modified ring sugars in the catalog"
        for smiles, value in MODIFIED_RING:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"unparseable catalog SMILES: {smiles}"
            result = recognize_sugar_skeleton(mol, anomeric_idx=None)
            assert result is None, (
                f"deriver returned a tuple {result!r} for out-of-scope sugar "
                f"{value!r} ({smiles}) — must fail closed to None (D-06)"
            )

    def test_none_mol_returns_none(self):
        """A None mol is handled gracefully (guard)."""
        assert recognize_sugar_skeleton(None, anomeric_idx=None) is None


@pytest.mark.unit
class TestSkeletonIdealization:
    """Phase 183 (WSC-04, D-01): physical idealize-then-lookup reproduces the
    clean parent skeleton for each modified-sugar class (deoxy / uronic / amino).

    This is the floor the Wave-1 generalization must defend: the raw modified
    fingerprint does NOT equal the clean parent (Pitfall 1 — the modification
    re-ranks ring CIP), so idealization MUST be physical (edit the molecule, then
    re-run rdCIPLabeler). Each idealization restores the EXACT canonical SMILES of
    clean beta-D-glucopyranose, and the existing deriver recovers (beta, D,
    glucopyranose) from it (RESEARCH §1).
    """

    # Verified modified-sugar SMILES (OPSIN-RT True this session); the clean parent.
    _DEOXY = "C[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
    _URONIC = "O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
    _AMINO = "N[C@@H]1[C@@H](O)[C@H](O)O[C@H](CO)[C@H]1O"
    _CLEAN = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"

    @staticmethod
    def _idealize(smiles):
        """Physical idealization (RESEARCH §1): restore the parent skeleton.

        deoxy -> add an exocyclic O on the bare terminal ring-attached CH3;
        uronic -> remove the carbonyl =O so COOH -> CH2OH;
        amino  -> SetAtomicNum(8) on the single nitrogen.
        Returns the canonical SMILES of the idealized (NEW) molecule.
        """
        m = Chem.RWMol(Chem.MolFromSmiles(smiles))
        has_n = any(a.GetSymbol() == "N" for a in m.GetAtoms())
        has_carbonyl = any(
            b.GetBondType() == Chem.BondType.DOUBLE
            and {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"C", "O"}
            for b in m.GetBonds()
        )
        if has_n:
            for a in m.GetAtoms():
                if a.GetSymbol() == "N":
                    a.SetAtomicNum(8)
                    break
        elif has_carbonyl:
            cc = next(
                a.GetIdx()
                for a in m.GetAtoms()
                if a.GetSymbol() == "C"
                and sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "O") == 2
            )
            dbl = next(
                b.GetOtherAtom(m.GetAtomWithIdx(cc)).GetIdx()
                for b in m.GetAtomWithIdx(cc).GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE
            )
            m.RemoveAtom(dbl)
        else:
            methyl = next(
                a.GetIdx()
                for a in m.GetAtoms()
                if a.GetSymbol() == "C" and a.GetDegree() == 1
            )
            o = m.AddAtom(Chem.Atom(8))
            m.AddBond(methyl, o, Chem.BondType.SINGLE)
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)

    def test_idealization_reproduces_clean_parent(self):
        """deoxy/uronic/amino each idealize to clean beta-D-glucopyranose (D-01)."""
        clean = Chem.CanonSmiles(self._CLEAN)
        for smi in (self._DEOXY, self._URONIC, self._AMINO):
            assert self._idealize(smi) == clean, (
                f"idealization of {smi} did not reproduce clean beta-D-glucopyranose"
            )

    def test_idealized_fingerprint_recovers_gluco(self):
        """The idealized parent recovers (beta, D, glucopyranose) via the deriver."""
        for smi in (self._DEOXY, self._URONIC, self._AMINO):
            ideal = Chem.MolFromSmiles(self._idealize(smi))
            recovered = recognize_sugar_skeleton(ideal, anomeric_idx=None) or lookup_sugar(
                Chem.MolToSmiles(ideal)
            )
            assert recovered == ("beta", "D", "glucopyranose"), (
                f"idealized {smi} recovered {recovered!r}, expected gluco"
            )


@pytest.mark.unit
class TestGlycosideClassName:
    """D-07: -ose -> -oside with an ASCII alpha/beta-D/L- prefix."""

    def test_glycoside_class_name_transform(self):
        assert (
            sugar_to_glycoside_class_name("beta", "D", "glucopyranose")
            == "beta-D-glucopyranoside"
        )
        assert (
            sugar_to_glycoside_class_name("alpha", "L", "rhamnopyranose")
            == "alpha-L-rhamnopyranoside"
        )
        assert (
            sugar_to_glycoside_class_name("beta", "D", "fructofuranose")
            == "beta-D-fructofuranoside"
        )

    def test_glycoside_class_name_is_ascii(self):
        """Descriptors stay ASCII — gold normalize() does NOT transliterate (Pitfall 6)."""
        name = sugar_to_glycoside_class_name("beta", "D", "glucopyranose")
        assert "β" not in name  # no Greek beta
        assert name.startswith("beta-D-")

    def test_glycoside_class_name_no_descriptors(self):
        """Empty anomer/config yields the bare -oside head (no leading hyphens)."""
        assert (
            sugar_to_glycoside_class_name("", "", "glucopyranose")
            == "glucopyranoside"
        )


@pytest.mark.unit
class TestSystematicMonosaccharideFamilyCoverage:
    """v23 CARB-02: the systematic monosaccharide family must be COMPLETE over the
    full ALL_SUGAR_NAMES union — every family base has all six
    {D,L} x {alpha,beta,unspecified-anomer} combos — with only the two
    chemically-impossible exclusions.  Guards against future catalog drift and
    against the coverage-asymmetry the PIN audit flagged (which was a false
    positive: the alpha-D/beta-D forms of the natural sugars live in the
    hand-curated SUGAR_RETAINED_NAMES, so completeness only holds over the union,
    NOT over SYSTEMATIC_MONOSACCHARIDE_NAMES alone).
    """

    # 2-ketopentoses (ribulo/xylulo) form only furanoses (a pyranose would need a
    # C6 the 5-carbon ketose lacks); aldotetroses (erythro/threo) likewise have no
    # pyranose.  These are the ONLY legitimate exclusions (chemistry, not data).
    _FAMILY_BASES = [
        # aldohexoses (pyranose + furanose)
        "allopyranose", "altropyranose", "glucopyranose", "mannopyranose",
        "gulopyranose", "idopyranose", "galactopyranose", "talopyranose",
        "allofuranose", "altrofuranose", "glucofuranose", "mannofuranose",
        "gulofuranose", "idofuranose", "galactofuranose", "talofuranose",
        # aldopentoses (pyranose + furanose)
        "arabinopyranose", "lyxopyranose", "ribopyranose", "xylopyranose",
        "arabinofuranose", "lyxofuranose", "ribofuranose", "xylofuranose",
        # aldotetroses (furanose only)
        "erythrofuranose", "threofuranose",
        # 2-ketohexoses (pyranose + furanose)
        "fructopyranose", "psicopyranose", "sorbopyranose", "tagatopyranose",
        "fructofuranose", "psicofuranose", "sorbofuranose", "tagatofuranose",
        # 2-ketopentoses (furanose only)
        "ribulofuranose", "xylulofuranose",
    ]
    _WANT = {("D", "alpha"), ("D", "beta"), ("D", ""),
             ("L", "alpha"), ("L", "beta"), ("L", "")}

    def test_every_family_base_has_full_dl_anomer_coverage(self):
        coverage = {}
        for _smi, (anomer, config, base) in ALL_SUGAR_NAMES.items():
            coverage.setdefault(base, set()).add((config, anomer))
        missing = {
            base: sorted(self._WANT - coverage.get(base, set()))
            for base in self._FAMILY_BASES
            if self._WANT - coverage.get(base, set())
        }
        assert not missing, (
            "systematic monosaccharide family incomplete over ALL_SUGAR_NAMES: "
            f"{missing}"
        )
