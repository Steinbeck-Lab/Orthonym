"""P-102.6.1.2 *O*-Glycosyl compounds: the glycosidic oxygen is expressed ONCE.

Blue Book **P-102.6.1.2 "*O*-Glycosyl compounds"** (``BlueBookV2/BlueBookV2.md:53915``):

    "The substituent group formed by removal of a hydrogen atom from the anomeric
    -OH group is considered as a compound substituent group formed by the
    'glycosyl' group and an 'oxy' group."

and its worked example (``:53927``)::

    1-[4-(beta-D-glucopyranosyloxy)phenyl]ethan-1-one
      (not 4-acetylphenyl beta-D-glucopyranoside;
       a ketone is senior to a hydroxy compound)

The aglycone is the PARENT, the sugar is a detachable prefix cited at the
aglycone's attachment locant, and the glycosylated oxygen appears only inside
that prefix -- the parent must not also cite it as a hydroxy.

Before this class was built, the decomposition fallback glued the sugar prefix
onto the aglycone's *alcohol* name, so the glycosidic oxygen was expressed twice
and the prefix carried no locant.
"""
import pytest
from rdkit import Chem

from orthonym.data.sugar_names import glycosyl_substituent_prefix


def _name(smiles):
    from orthonym import Orthonym
    return Orthonym().name(smiles)


# --------------------------------------------------------------------------
# The oxygen is counted ONCE, and the prefix carries the aglycone's locant
# --------------------------------------------------------------------------

@pytest.mark.unit
class TestGlycosidicOxygenCountedOnce:
    """The defect these guard: '(beta-D-glucopyranosyloxy)...-3,5,7-trihydroxy-'
    cited the 7-O both as a hydroxy on the aglycone and inside the prefix."""

    def test_flavan_glucoside_locanted_and_single_counted(self):
        """The 7-O bears the sugar, so only 3 and 5 remain as hydroxy."""
        name = _name(
            "OC[C@H]1O[C@@H](Oc2cc(O)c3c(c2)O[C@@H](c2ccc(O)cc2)[C@H](O)C3)"
            "[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name == (
            "(2S,3R)-7-(beta-D-glucopyranosyloxy)-3,5-dihydroxy-"
            "2-(4-hydroxyphenyl)-3,4-dihydro-2H-1-benzopyran"
        )
        # The oxygen is not double-counted: 'trihydroxy' was the bug.
        assert "trihydroxy" not in name

    def test_glycosyloxy_prefix_carries_a_locant(self):
        """A prefix with no attachment locant is what OPSIN either refuses or
        attaches wrongly; the BB cites '4-(...)' explicitly."""
        name = _name("O=Cc1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1")
        assert name == "4-(beta-D-glucopyranosyloxy)benzaldehyde"
        assert not name.startswith("(beta-D-glucopyranosyloxy)")

    def test_benzaldehyde_aglycone_keeps_all_sugar_oxygens(self):
        """The skeleton-only '(oxan-2-yl)oxy' fallback discarded the sugar's four
        hydroxy groups and its CH2OH while reporting the whole fragment as
        covered, so nothing downstream could see the loss."""
        name = _name("O=Cc1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1")
        assert "oxan-2-yl" not in name
        assert "oxolan-2-yl" not in name

    def test_bluebook_worked_example(self):
        """BB :53927 verbatim. CHARACTERIZATION, not a regression guard for this
        change: measured to already hold before it (HEAD 62c77fa0)."""
        name = _name("CC(=O)c1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1")
        assert name == "1-[4-(beta-D-glucopyranosyloxy)phenyl]ethan-1-one"


# --------------------------------------------------------------------------
# The perception primitive: what it names and what it refuses
# --------------------------------------------------------------------------

def _sugar_rings(mol):
    out = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue
        syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
        if syms.count("O") != 1 or any(s not in ("C", "O") for s in syms):
            continue
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        out.append(ring)
    return out


def _glyco_fragment(smiles):
    """Return (mol, frag_atoms, o_idx) for the AGLYCONE-PROXIMAL glycosidic O.

    The far side of the chosen linkage must not itself be a sugar ring atom --
    otherwise a disaccharide's INNER (sugar->sugar) linkage is picked and the
    fragment is a single monosaccharide, which is not the case under test.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    rings = _sugar_rings(mol)
    all_sugar = set().union(*rings) if rings else set()
    for ring in rings:
        ring_o = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O"][0]
        for i in ring:
            a = mol.GetAtomWithIdx(i)
            if a.GetSymbol() != "C":
                continue
            if ring_o not in [n.GetIdx() for n in a.GetNeighbors()]:
                continue
            for nb in a.GetNeighbors():
                if nb.GetIdx() in ring or nb.GetSymbol() != "O":
                    continue
                far = [n.GetIdx() for n in nb.GetNeighbors() if n.GetIdx() != i]
                if len(far) != 1 or far[0] in all_sugar:
                    continue
                seen = {nb.GetIdx()}
                stack = [i]
                while stack:
                    cur = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    for n in mol.GetAtomWithIdx(cur).GetNeighbors():
                        if n.GetIdx() not in seen:
                            stack.append(n.GetIdx())
                if far[0] in seen:
                    continue
                return mol, seen, nb.GetIdx()
    raise AssertionError(f"no glycosidic linkage found in {smiles}")


@pytest.mark.unit
class TestGlycosylSubstituentPrefix:

    def test_names_a_plain_retained_glycosyl(self):
        mol, frag, o = _glyco_fragment(
            "O=Cc1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1")
        assert glycosyl_substituent_prefix(mol, frag, o) == \
            "beta-D-glucopyranosyloxy"

    def test_anomer_and_configuration_are_preserved(self):
        """alpha/beta and D/L come from the fragment's own stereo, so a different
        sugar must not borrow another's descriptors."""
        mol, frag, o = _glyco_fragment(
            "C[C@@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@H](O)[C@H]1O")
        assert glycosyl_substituent_prefix(mol, frag, o) == \
            "alpha-L-rhamnopyranosyloxy"

    def test_refuses_a_uronic_glycosyl(self):
        """BB :53929 cites a uronic glycosyl as 'beta-D-glucopyranosyluronic
        acid', not as a '...osyloxy' token; the naive contraction
        'glucuronopyranosyloxy' is a fabricated morpheme OPSIN cannot parse."""
        mol, frag, o = _glyco_fragment(
            "O=C(O)C1OC(Oc2ccc3ccc(=O)oc3c2O)C(O)C(O)C1O")
        assert glycosyl_substituent_prefix(mol, frag, o) is None

    def test_refuses_an_oligosaccharide_substituent(self):
        """A disaccharide substituent needs the P-102 (1->n) linkage notation
        INSIDE the prefix, which this primitive does not build."""
        mol, frag, o = _glyco_fragment(
            "C[C@@H]1O[C@@H](O[C@H]2[C@H](Oc3cc(O)c4c(c3)O[C@H](c3ccccc3)CC4=O)"
            "O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O")
        assert glycosyl_substituent_prefix(mol, frag, o) is None

    def test_refuses_when_the_attachment_atom_is_not_the_glycosidic_oxygen(self):
        """A C-glycosyl (P-102.6.1.4) is a different construction: no 'oxy'."""
        mol, frag, o = _glyco_fragment(
            "O=Cc1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1")
        anomeric = [n.GetIdx() for n in mol.GetAtomWithIdx(o).GetNeighbors()
                    if n.GetIdx() in frag][0]
        assert glycosyl_substituent_prefix(mol, frag, anomeric) is None

    def test_refuses_a_uronic_glycosyl_that_HAS_anomer_and_config(self):
        """Isolating witness for the uronic guard.

        The coumarin glucuronide in ``test_refuses_a_uronic_glycosyl`` looks up as
        ``('', '', 'glucuronopyranose')`` -- no anomer, no configuration -- so the
        anomer/config guard would refuse it even with the uronic guard removed
        (measured: that mutation SURVIVED). This stereo-defined glucuronide looks
        up as ``('beta', 'D', 'glucuronopyranose')``, so the uronic guard is the
        only thing that can refuse it.
        """
        mol, frag, o = _glyco_fragment(
            "O=C(O)[C@H]1O[C@@H](Oc2ccc(C=O)cc2)[C@H](O)[C@@H](O)[C@@H]1O")
        assert glycosyl_substituent_prefix(mol, frag, o) is None

    def test_refuses_a_sugar_with_no_anomeric_or_configurational_descriptor(self):
        """Isolating witness for the anomer/config guard.

        Non-uronic and undecorated, so neither of those guards applies; it looks
        up as ``('', '', 'glucopyranose')``. ``sugar_to_glycosyloxy_prefix`` drops
        both descriptors together when either is missing, which would spell a
        stereochemically unspecified group.
        """
        mol, frag, o = _glyco_fragment(
            "O=Cc1ccc(OC2OC(CO)C(O)C(O)C2O)cc1")
        assert glycosyl_substituent_prefix(mol, frag, o) is None

    def test_refuses_a_decorated_glycosyl(self):
        """Isolating witness for the decorated-base guard.

        N-acetylglucosamine looks up as
        ``('beta', 'D', '2-acetamido-2-deoxy-glucopyranose')`` -- non-uronic, with
        both descriptors -- so only the decorated-base guard can refuse it.
        Concatenating would give 'beta-D-2-acetamido-2-deoxy-glucopyranosyloxy',
        whereas P-102.6.1.2 (:53935) requires the decorated glycosyl inside its
        own enclosing marks before 'oxy'.
        """
        mol, frag, o = _glyco_fragment(
            "O=Cc1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2NC(C)=O)cc1")
        assert glycosyl_substituent_prefix(mol, frag, o) is None

    def test_refuses_a_thioglycoside(self):
        """An S-linked glycoside is not an *O*-glycosyl compound and must never be
        spelled with the 'oxy' morpheme (that would name a different molecule).

        CONTRACT test, not a guard-isolating witness: measured by mutation, the
        explicit oxygen-attach check is SUBSUMED by the fragment->sugar lookup
        (an S-capped fragment canonicalises to a thiol, which no sugar entry
        matches), so deleting that check alone -- or together with the
        ring-containment check -- does not change this result. The check is
        defence in depth for the 'oxy' morpheme's meaning, not dead logic; this
        test pins the BEHAVIOUR, which is what callers depend on.
        """
        mol = Chem.MolFromSmiles(
            "O=Cc1ccc(S[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1")
        s_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S"][0]
        aromatic = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
        frag, stack = set(), [s_idx]
        while stack:
            cur = stack.pop()
            if cur in frag or cur in aromatic:
                continue
            frag.add(cur)
            for n in mol.GetAtomWithIdx(cur).GetNeighbors():
                if n.GetIdx() not in frag and n.GetIdx() not in aromatic:
                    stack.append(n.GetIdx())
        assert glycosyl_substituent_prefix(mol, frag, s_idx) is None

    def test_refuses_a_fragment_that_does_not_hold_the_whole_sugar_ring(self):
        """If the caller hands over a fragment that cuts through the sugar ring,
        the fragment is not a monosaccharide and must not be named as one.

        CONTRACT test, not a guard-isolating witness: like the thioglycoside case,
        the ring-containment check is subsumed by the fragment->sugar lookup (a
        truncated ring canonicalises to something no sugar entry matches), proven
        by a double mutation that removes both subsumed checks and still passes.
        """
        mol, frag, o = _glyco_fragment(
            "O=Cc1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1")
        ring = next(r for r in _sugar_rings(mol) if set(r) <= frag)
        truncated = set(frag) - {ring[-1]}
        assert glycosyl_substituent_prefix(mol, truncated, o) is None

    def test_refuses_a_non_sugar_oxygen_ring(self):
        """An oxane that is not a monosaccharide must not be named as one."""
        mol = Chem.MolFromSmiles("O=Cc1ccc(OC2OCCCC2)cc1")
        o_idx = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "O" and not a.IsInRing()
                 and a.GetDegree() == 2][0]
        frag = set()
        stack = [o_idx]
        aromatic = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
        while stack:
            cur = stack.pop()
            if cur in frag or cur in aromatic:
                continue
            frag.add(cur)
            for n in mol.GetAtomWithIdx(cur).GetNeighbors():
                if n.GetIdx() not in frag and n.GetIdx() not in aromatic:
                    stack.append(n.GetIdx())
        assert glycosyl_substituent_prefix(mol, frag, o_idx) is None


# --------------------------------------------------------------------------
# Negatives: the sugar->sugar (oligosaccharide, P-102) path is untouched
# --------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C[C@@H]1O[C@@H](O[C@@H]2[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]2O)"
     "[C@@H](O)[C@H](O)[C@@H]1O",
     "alpha-L-fucopyranosyl-(1->2)-alpha-D-glucopyranose"),
    ("OC[C@H]1O[C@H](O[C@@H]2C(O)O[C@H](CO)[C@@H](O)[C@@H]2O)"
     "[C@@H](O)[C@@H](O)[C@@H]1O",
     "alpha-D-mannopyranosyl-(1->2)-D-mannopyranose"),
    ("OC[C@H]1O[C@@H](O[C@@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)"
     "[C@@H](O)[C@@H](O)[C@@H]1O",
     "beta-D-mannopyranosyl-(1->4)-beta-D-galactopyranose"),
])
def test_sugar_to_sugar_linkage_notation_unchanged(smiles, expected):
    """A glycosidic bond between two SUGARS keeps the P-102 (1->n) form. The
    glycosyloxy prefix must not capture it -- these are the discriminator."""
    assert _name(smiles) == expected
