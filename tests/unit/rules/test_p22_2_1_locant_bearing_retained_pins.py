""": Tables 2.2 and 2.3 make the LOCANT-BEARING form the PIN.

Task AA5, the sibling of Task AA3 (``test_p22_2_1_pyran_indicated_hydrogen.py``)
and the same root cause: a bare retained stem imported from
``data/opsin_imports/aryl_groups.py`` was promoted into ``ALL_RETAINED_NAMES``
and short-circuited the dispatch before any rule ran.

Section ** "Retained names of heteromonocycles"** (heading
``the Blue Book``); its lead-in at ``:8117`` reads *"Retained names for
saturated heteromonocycles are given in Table 2.3."* The governing lines,
verbatim::

    :8137 isoxazole 1,2-oxazole (PIN) isothiazole (S instead of O)
           1,2-thiazole (PIN) isoselenazole (Se instead of O) 1,2-selenazole
           (PIN) isotellurazole (Te instead of O) 1,2-tellurazole (PIN)
    :8165 selenophene (PIN)
    :8170 tellurophene (PIN)
    :8182 oxazolidine 1,3-oxazolidine (PIN) thiazolidine (S instead of O)
           1,3-thiazolidine (PIN) selenazolidine (Se instead of O)
    :8184 1,2-oxazolidine (PIN) isothiazolidine (S instead of O)
           1,2-thiazolidine (PIN) isoselenazolidine (Se instead of O)
           1,2-selenazolidine (PIN) isotellurazolidine (Te instead of O)

Table 2.3 is OCR'd as interleaved two-column text. Only two pairings there need
reconstruction (``isoxazolidine`` at:8180 ->:8184, and ``isotellurazolidine``
at:8184 ->:8188); both were cross-checked against
<https://iupac.qmul.ac.uk/BlueBook/> and are independently forced by:8137, where
the identical iso-/1,2- mapping is printed for the mancude analogues on a single
non-interleaved line. Every other row asserted here sits on one line.

⚠ **No round-trip oracle can protect this class.** OPSIN 2.9.0 resolves the bare
``thiazolidine``, ``isothiazolidine`` and ``selenofuran`` to exactly the same
structures as their locant-bearing PINs, so name -> structure -> InChIKey agrees
for names the Blue Book prints as not-the-PIN. These are *spelling* assertions,
checked against the Blue Book by eye. That is the point of the file.
"""
import pytest

from orthonym.data import ALL_RETAINED_NAMES
from orthonym.namer import name_compound


# --------------------------------------------------------------------------
# 1. The nine parents that were emitting a non-PIN string
# --------------------------------------------------------------------------

PARENTS = [
    ("c1cc[se]c1", "selenophene"),           #:8165
    ("c1cc[te]c1", "tellurophene"),          #:8170
    ("c1cn[se]c1", "1,2-selenazole"),        #:8137
    ("c1cn[te]c1", "1,2-tellurazole"),       #:8137
    ("C1CSCN1", "1,3-thiazolidine"),         #:8182
    ("C1CNOC1", "1,2-oxazolidine"),          #:8180 +:8184
    ("C1CNSC1", "1,2-thiazolidine"),         #:8184
    ("C1CN[Se]C1", "1,2-selenazolidine"),    #:8184
    ("C1CN[Te]C1", "1,2-tellurazolidine"),   #:8184 +:8188
]


@pytest.mark.parametrize("smiles,expected", PARENTS)
def test_parent_ring_emits_the_locant_bearing_pin(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# 2. Substituted derivatives -- the actual bulk of the defect
# --------------------------------------------------------------------------

# The parents are 9 molecules; the derivatives are unbounded. They inherit the
# defect through a DIFFERENT producer: the parent lookup for a substituted ring
# is ``data/retained_names.py::get_retained_name`` (feeding
# ``rules/heterocycles.py::name_heterocycle``), not the whole-molecule dict
# lookup in ``routing/dispatch_table.py::_handle_retained_name`` that serves the
# parents. Both read ``ALL_RETAINED_NAMES``, which is why one deny row fixes
# both -- verified by a cold trace, one fresh process per molecule.
SUBSTITUTED = [
    ("C1CSCN1C", "3-methyl-1,3-thiazolidine"),
    ("CC1CSCN1", "4-methyl-1,3-thiazolidine"),
    ("C1CSCN1c1ccccc1", "3-phenyl-1,3-thiazolidine"),
    ("C1CNSC1C", "5-methyl-1,2-thiazolidine"),
    ("CN1CCCS1", "2-methyl-1,2-thiazolidine"),
    ("C1CNOC1C", "5-methyl-1,2-oxazolidine"),
    ("CN1CCCO1", "2-methyl-1,2-oxazolidine"),
    ("O=C1CCON1", "3-oxo-1,2-oxazolidine"),
    ("Cc1cc[se]c1", "3-methylselenophene"),
    ("OC(=O)c1cc[se]c1", "selenophene-3-carboxylic acid"),
    ("Cc1cc[te]c1", "3-methyltellurophene"),
    ("Cc1cn[se]c1", "4-methyl-1,2-selenazole"),
    ("C1CN[Se]C1C", "5-methyl-1,2-selenazolidine"),
    ("C1CN[Te]C1C", "5-methyl-1,2-tellurazolidine"),
]


@pytest.mark.parametrize("smiles,expected", SUBSTITUTED)
def test_substituted_derivative_inherits_the_pin(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# 3. The suffixed forms were ALREADY right -- guard against double citation
# --------------------------------------------------------------------------

# ``rules/heterocycles.py::_retained_heteroatom_locant_prefix``
# already injected ``1,3-`` for a suffixed thiazolidine, which is why
# ``1,3-thiazolidin-4-one`` has always been correct while ``3-methylthiazolidine``
# was wrong. That injector keys on the BARE stem and returns '' on a miss, so
# after this deny it silently no-ops instead of producing ``1,3-1,3-``. These
# three are the regression guard for that interaction; all three are also rows in
# ``benchmarks/the gold set/packs/characteristic_groups.json``.
ALREADY_CORRECT_SUFFIXED = [
    ("O=C1CSCN1", "1,3-thiazolidin-4-one"),
    ("OC1CSCN1", "1,3-thiazolidin-4-ol"),
    ("OC(=O)C1CSCN1", "1,3-thiazolidine-4-carboxylic acid"),
    ("O=C1CSC(=S)N1", "2-sulfanylidene-1,3-thiazolidin-4-one"),
]


@pytest.mark.parametrize("smiles,expected", ALREADY_CORRECT_SUFFIXED)
def test_suffixed_form_is_not_double_locanted(smiles, expected):
    name = name_compound(smiles)
    assert "1,3-1,3-" not in name
    assert name == expected


# --------------------------------------------------------------------------
# 3b. The ring AS A SUBSTITUENT -- where withdrawing the stem broke things
# --------------------------------------------------------------------------

# These are the regression the deny rows caused and
# ``ring_substituents.py::_bare_stem_was_withdrawn_as_non_pin`` repairs.
# ``identify_ring_system`` has no branch for a two-heteroatom saturated
# five-ring, so before Task AA5 the substituent path's ONLY answer for these
# rings was the retained bare stem. Withdrawing it turned ten correct names
# into ``unknown organic compound`` -- textbook CLAUDE.md a project rule. Every
# expectation below was confirmed against OPSIN 2.9.0: each name round-trips to
# exactly the input SMILES.
RING_AS_SUBSTITUENT = [
    ("OCC1CSCN1", "(1,3-thiazolidin-4-yl)methanol"),
    ("OCCN1CCSC1", "2-(1,3-thiazolidin-3-yl)ethan-1-ol"),
    ("OCC1CCNS1", "(1,2-thiazolidin-5-yl)methanol"),
    ("OCCN1CCCS1", "2-(1,2-thiazolidin-2-yl)ethan-1-ol"),
    ("OCC1CCNO1", "(1,2-oxazolidin-5-yl)methanol"),
    ("OCCN1CCCO1", "2-(1,2-oxazolidin-2-yl)ethan-1-ol"),
    ("OCC1CCN[Se]1", "(1,2-selenazolidin-5-yl)methanol"),
    ("OCc1cc[se]c1", "(selenophen-3-yl)methanol"),
    ("OCCc1cc[se]c1", "2-(selenophen-3-yl)ethan-1-ol"),
    ("OCc1cn[se]c1", "(1,2-selenazol-4-yl)methanol"),
]


@pytest.mark.parametrize("smiles,expected", RING_AS_SUBSTITUENT)
def test_ring_as_substituent_keeps_a_name(smiles, expected):
    """Must be the PIN -- and above all must NOT be an abstention."""
    name = name_compound(smiles)
    assert "unknown" not in name, (
        "withdrawing a non-PIN stem must fall through to the systematic PIN, "
        f"never to an abstention: {smiles} -> {name}"
    )
    assert name == expected


# The fallback must stay narrow. Generalising it to "any heteromonocycle the
# retained table misses" newly names 1660 substituent stems across the corpora
# and is wrong on many -- macrocyclic free valences land on a ring oxygen.
# These rings are NOT deny-listed, so the fallback must not fire for them and
# whatever the other producers say must be unchanged.
def test_fallback_does_not_fire_for_rings_that_were_never_denied():
    from rdkit import Chem
    from orthonym.rules.ring_substituents import (
        _bare_stem_was_withdrawn_as_non_pin,
    )

    for smi in ("C1CCCO1", "C1CCCN1", "c1ccsc1", "c1ccoc1", "C1COCCN1"):
        frag = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        assert not _bare_stem_was_withdrawn_as_non_pin(frag), smi
    #...and DOES fire for the ones this task withdrew (guards the guard: a
    # predicate that always returns False would pass the loop above).
    for smi in ("C1CSCN1", "C1CNSC1", "C1CNOC1"):
        frag = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        assert _bare_stem_was_withdrawn_as_non_pin(frag), smi


# --------------------------------------------------------------------------
# 4. The root cause, guarded structurally
# --------------------------------------------------------------------------

# ``aryl_groups.py`` is a generated OPSIN import: every row carries
# ``is_pin: False`` as a hard-coded generator default, so a re-import can
# silently re-add these rows. Only the adjudicated deny list keeps them out.
# Note ``selenofuran``/``tellurofuran`` do not appear in the Blue Book at all --
# they are OPSIN parser synonyms, and ``_select_primary_name`` takes ``names[0]``,
# so for ``['selenofuran', 'selenophene']`` the synonym won and the PIN was
# discarded.
BARE_NON_PIN_STEMS = {
    "selenofuran", "tellurofuran", "isoselenazole", "isotellurazole",
    "thiazolidine", "isoxazolidine", "isothiazolidine",
    "isoselenazolidine", "isotellurazolidine",
}


def test_no_bare_non_pin_stem_is_a_headline_retained_name():
    offenders = {
        smi: name for smi, name in ALL_RETAINED_NAMES.items()
        if name.lower().strip() in BARE_NON_PIN_STEMS
    }
    assert offenders == {}, (
        "P-22.2.1 'Retained names of heteromonocycles' (BlueBookV2.md:8109) "
        "prints the locant-bearing form as the PIN for each of these; the bare "
        f"stem must not be reachable as a whole-molecule PIN: {offenders}"
    )


def test_denied_stems_remain_reachable_under_trivial_style():
    """The deny is a demotion, not a deletion.

    ``pin_policy.py``'s contract is that a denied name "leaves the PIN-path
    lookup and moves to the general-only companion dict, reachable via
    ``--trivial``". If a future change deletes the rows outright instead of
    denying them, this fails -- and the general-nomenclature names, which are
    perfectly legitimate, would be silently lost.
    """
    from orthonym.data import GENERAL_RETAINED_NAMES

    demoted = {
        name.lower().strip() for name in GENERAL_RETAINED_NAMES.values()
    }
    missing = BARE_NON_PIN_STEMS - demoted
    assert missing == set(), f"denied stems lost rather than demoted: {missing}"


# --------------------------------------------------------------------------
# 5. Controls: the members of the same tables that were ALREADY correct
# --------------------------------------------------------------------------

# 2 of the 4 members of the 1,2-azole series and 3 of the 4 members of the
# 1,3-azolidine series were already emitting the PIN before this change. That
# is what makes the defect a data problem rather than a rule problem: the same
# generic path serves all of them. These must not move.
ALREADY_CORRECT_CONTROLS = [
    ("c1cnoc1", "1,2-oxazole"),
    ("c1cnsc1", "1,2-thiazole"),
    ("c1cocn1", "1,3-oxazole"),
    ("c1cscn1", "1,3-thiazole"),
    ("C1COCN1", "1,3-oxazolidine"),
    ("C1C[Se]CN1", "1,3-selenazolidine"),
    ("C1C[Te]CN1", "1,3-tellurazolidine"),
    ("c1ccsc1", "thiophene"),
    ("c1ccoc1", "furan"),
    ("C1CCNC1", "pyrrolidine"),
    ("C1COCCN1", "morpholine"),
    ("C1CNNC1", "pyrazolidine"),
    ("C1CNCN1", "imidazolidine"),
]


@pytest.mark.parametrize("smiles,expected", ALREADY_CORRECT_CONTROLS)
def test_already_correct_table_members_do_not_move(smiles, expected):
    assert name_compound(smiles) == expected
