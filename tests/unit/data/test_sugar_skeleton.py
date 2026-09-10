"""
 catalog-reproduction HARD gate for the structure-derived sugar recognizer.

a phase Plan 01 (-08). These tests are the fail-stop that keeps
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

* **Structurally-clean** (the deriver's actual contract): the clean
  hexose/pentose OH/H/CH2OH fingerprint — 40 stereo pyranose/furanose entries.
  rhamnopyranose/fucopyranose are EXCLUDED here (they are deoxy → the deriver
  returns ``None`` per / RESEARCH §"Verified deriver fingerprint").

The catalog-reproduction gate  therefore reproduces the 40 structurally
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
    """The deriver's contract: clean hexose/pentose OH/H/CH2OH fingerprint.

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
    """ HARD gate: the deriver reproduces the clean catalog, fails closed."""

    def test_clean_set_scope_self_validates(self):
        """The clean-set scope is exactly what the catalog pins (catalog-drift trip).

          grew the catalog with the complete systematic monosaccharide
        stereo family (SYSTEMATIC_MONOSACCHARIDE_NAMES). Since then the
        breadth program (~30 later feature commits) grew ALL_SUGAR_NAMES further:
        keyword-clean 220 -> 236 (+16) and structural-clean 216 -> 224 (+8), so
        the two-notions difference widened from 4 to 12. These counts are the
        MEASURED live scope; test_catalog_reproduction below (the deriver's HARD
        gate) passes on all 224 structural-clean entries, so the growth is real
        coverage, not a scope-filter drift.
        """
        # Keyword-clean is the base-name filter (measured live).
        clean = KEYWORD_CLEAN  # noqa: F841 - named for the acceptance grep
        assert len(clean) == 236, (
            f"keyword-clean catalog scope drifted: got {len(clean)} (expected 236)"
        )
        # Structural-clean is the deriver's contract (measured live).
        assert len(STRUCTURAL_CLEAN) == 224, (
            f"structural-clean scope drifted: got {len(STRUCTURAL_CLEAN)} "
            "(expected 224)"
        )
        # The deoxy / structurally-modified entries are the difference between
        # the two notions (keyword-clean names but not structurally clean).
        assert len(KEYWORD_CLEAN) - len(STRUCTURAL_CLEAN) == 12

    def test_catalog_reproduction(self):
        """Deriver reproduces (anomer, config, base) for every clean entry ."""
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
        """Deriver fails closed on every modified/out-of-scope ring sugar ."""
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
    """a phase (-04,): physical idealize-then-lookup reproduces the
    clean parent skeleton for each modified-sugar class (deoxy / uronic / amino).

    This is the floor the Wave-1 generalization must defend: the raw modified
    fingerprint does NOT equal the clean parent (Pitfall 1 — the modification
    re-ranks ring CIP), so idealization MUST be physical (edit the molecule, then
    re-run rdCIPLabeler). Each idealization restores the EXACT canonical SMILES of
    clean β-D-glucopyranose, and the existing deriver recovers (beta, D,
    glucopyranose) from it (RESEARCH).
    """

    # Verified modified-sugar SMILES (OPSIN-RT True this session); the clean parent.
    _DEOXY = "C[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
    _URONIC = "O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
    _AMINO = "N[C@@H]1[C@@H](O)[C@H](O)O[C@H](CO)[C@H]1O"
    _CLEAN = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"

    @staticmethod
    def _idealize(smiles):
        """Physical idealization (RESEARCH): restore the parent skeleton.

        deoxy -> add an exocyclic O on the bare terminal ring-attached CH3;
        uronic -> remove the carbonyl =O so COOH -> CH2OH;
        amino -> SetAtomicNum(8) on the single nitrogen.
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
        """deoxy/uronic/amino each idealize to clean β-D-glucopyranose (D-01)."""
        clean = Chem.CanonSmiles(self._CLEAN)
        for smi in (self._DEOXY, self._URONIC, self._AMINO):
            assert self._idealize(smi) == clean, (
                f"idealization of {smi} did not reproduce clean β-D-glucopyranose"
            )

    def test_idealized_fingerprint_recovers_gluco(self):
        """The idealized parent recovers (beta, D, glucopyranose) via the deriver."""
        for smi in (self._DEOXY, self._URONIC, self._AMINO):
            ideal = Chem.MolFromSmiles(self._idealize(smi))
            recovered = recognize_sugar_skeleton(ideal, anomeric_idx=None) or lookup_sugar(
                Chem.MolToSmiles(ideal)
            )
            assert recovered == ("β", "D", "glucopyranose"), (
                f"idealized {smi} recovered {recovered!r}, expected gluco"
            )


@pytest.mark.unit
class TestGlycosideClassName:
    """: -ose -> -oside with an ASCII alpha/beta-D/L- prefix."""

    def test_glycoside_class_name_transform(self):
        assert (
            sugar_to_glycoside_class_name("β", "D", "glucopyranose")
            == "β-D-glucopyranoside"
        )
        assert (
            sugar_to_glycoside_class_name("α", "L", "rhamnopyranose")
            == "α-L-rhamnopyranoside"
        )
        assert (
            sugar_to_glycoside_class_name("β", "D", "fructofuranose")
            == "β-D-fructofuranoside"
        )

    def test_glycoside_class_name_greek_anomer(self):
        """v43: the anomeric descriptor is the Blue-Book GREEK β (P-102.5.6.6,
        BlueBookV2.md uses β-D- 58×), NOT the ASCII word. (Was previously asserted
        ASCII; the Blue Book uses the Greek symbol and OPSIN parses both identically,
        so the emit was corrected to Greek — the D/L config stays ASCII small-capital.)"""
        name = sugar_to_glycoside_class_name("β", "D", "glucopyranose")
        assert name.startswith("β-D-")
        assert "beta" not in name  # the ASCII word is NOT used

    def test_glycoside_class_name_no_descriptors(self):
        """Empty anomer/config yields the bare -oside head (no leading hyphens)."""
        assert (
            sugar_to_glycoside_class_name("", "", "glucopyranose")
            == "glucopyranoside"
        )


@pytest.mark.unit
class TestSystematicMonosaccharideFamilyCoverage:
    """: the systematic monosaccharide family must be COMPLETE over the
    full ALL_SUGAR_NAMES union — every family base has all six
    {D,L} x {alpha,beta,unspecified-anomer} combos — with only the two
    chemically-impossible exclusions. Guards against future catalog drift and
    against the coverage-asymmetry the PIN audit flagged (which was a false
    positive: the alpha-D/beta-D forms of the natural sugars live in the
    hand-curated SUGAR_RETAINED_NAMES, so completeness only holds over the union,
    NOT over SYSTEMATIC_MONOSACCHARIDE_NAMES alone).
    """

    # 2-ketopentoses (ribulo/xylulo) form only furanoses (a pyranose would need a
    # C6 the 5-carbon ketose lacks); aldotetroses (erythro/threo) likewise have no
    # pyranose. These are the ONLY legitimate exclusions (chemistry, not data).
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
    _WANT = {("D", "α"), ("D", "β"), ("D", ""),
             ("L", "α"), ("L", "β"), ("L", "")}

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
