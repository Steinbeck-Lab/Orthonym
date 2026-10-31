"""Silicon, germanium and boron centres that are the parent of an ion: the -ylium cation and the lambda -ide anion.

The centre is a mononuclear parent hydride (silane, germane, borane) and every group on it is a detachable
prefix, named by ``rules.centre_ligands`` (the substituent-prefix namer of the substitutive path, with the
atom-coverage veto) and cited as the groups of a mononuclear hydride are, the Blue Book).
Tin, lead, aluminium and gallium are metals of ``data/metal_scope.py``: admitting one to a mechanism is an edit
of that table followed by a panel measurement, and ``route_charged`` hands a metal on before it reaches here. Tin
and lead are admitted to ``PARENT_HYDRIDE_IONS`` (2026-10-30): their cations are named here as silicon's are.
"""
from rdkit import Chem

# "Cationic centers in parent hydrides" (the Blue Book), heading "Specific method": the
# cation formally derived by removal of a hydride ion from a mononuclear parent hydride of Group 14 takes
# 'ylium' for the 'ane' of the hydride's name -- "silane, SiH4, germane, GeH4, stannane, SnH4, and plumbane,
# PbH4... by replacing the 'ane' ending... by the suffix 'ylium'" ((C6H5)3Si+ triphenylsilylium (PIN),
#:41557). The general method of the same section (:41559, "added to the name of the parent hydride, with
# elision of the final 'e'") gives boron the same ending (boranylium).
# element -> (name of the cation, standard bonding number,
_GROUP_YLIUM = {'Si': ('silylium', 4), 'Ge': ('germylium', 4), 'B': ('boranylium', 3),
                # the metals of the same paragraph, read only where the table admits them (PARENT_HYDRIDE_IONS)
                'Sn': ('stannylium', 4), 'Pb': ('plumbylium', 4)}
_METAL_YLIUM = frozenset({'Sn', 'Pb'})

# "ANIONS FORMED BY ADDITION OF HYDRIDE IONS" (the Blue Book), method (2) (:41094): 'ide' on a
# parent hydride whose bonding number at the anionic site is higher than the standard one and is expressed by
# the lambda-convention ('tetramethyl-lambda5-phosphanide',:41106). element -> (parent hydride, standard
# bonding number).
_LAMBDA_IDE = {'Si': ('silane', 4), 'Ge': ('germane', 4)}


def emit_group_ylium(mol, center_idx: int) -> str:
    """ (the Blue Book, "Specific method"; general method:41559): name a mononuclear
    silicon, germanium or boron centre that has lost a hydride ion -- the +1 centre whose bonds and hydrogens
    make up one less than its standard bonding number, Table 1.3: 4 for Si and Ge, 3 for B) -- on
    the parent hydride's own cation, with every group on it a detachable prefix:

        (C6H5)3Si+ -> triphenylsilylium (PIN) [the Blue Book]

    Scope (fail closed): a centre in no ring, single bonds only, no radical electron (a centre with one bond
    too many is a radical cation,, not this cation), no neighbour of its own element nor one atom
    of a heteroatom between two of them (the parent hydride is then the longer chain, 'heptamethyltrisilan-2-
    ylium (PIN)', the Blue Book; ``radicals._centre_extends_to_chain``), and only for the molecule the outermost
    ``name`` call was given, never a fragment a nested frame derives from another input
    (``universal_halane.names_the_input``: a nested frame's label would lift the whole name). The caller
    reads the name back to the full InChIKey. Returns '' on any decline."""
    center = mol.GetAtomWithIdx(center_idx)
    entry = _GROUP_YLIUM.get(center.GetSymbol())
    if center.GetSymbol() in _METAL_YLIUM:
        from ..data.metal_scope import PARENT_HYDRIDE_IONS, has_mechanism
        if not has_mechanism(center.GetSymbol(), PARENT_HYDRIDE_IONS):
            return ''
    if entry is None or center.GetFormalCharge() != 1 or center.IsInRing() or center.GetNumRadicalElectrons():
        return ''
    suffix, standard = entry
    if (any(b.GetBondType() != Chem.BondType.SINGLE for b in center.GetBonds())
            or center.GetDegree() + center.GetTotalNumHs() != standard - 1
            or center.GetIsotope() or any(nb.GetAtomicNum() == 1 and nb.GetIsotope() for nb in center.GetNeighbors())):
        return ''
    from ..assembly.universal_halane import names_the_input
    from .radicals import _centre_extends_to_chain
    if not names_the_input(mol) or _centre_extends_to_chain(center):
        return ''
    from .centre_ligands import centre_name
    return centre_name(mol, center_idx, suffix, suffix)


def name_lambda_ide_anion(mol) -> str:
    """ method (2) (the Blue Book): the one charged atom of ``mol`` is a silicon or germanium
    centre AT its standard bonding number with the -1 charge and single bonds only -- R4Si-, a radical anion
    by its electron count on which RDKit counts no radical electron -- named on the lambda parent with
    'ide': 'tetramethyl-lambda5-silanide'. Method (1) gives the preferred names (:41099) and for this centre
    it is the radical-anion name of (:43424, 'trimethylboranuidyl (PIN)',:43430), 'tetramethyl-
    silanuidyl'; OPSIN reads that only under -r, and the read-back policy (``validation.read_back_policy``)
    gives -r to an input with radical electrons, of which RDKit counts none here. The lambda name is read by
    OPSIN to the input as it is, so it is the name shipped, below the PIN (the label of the pipeline, which
    records it as a non-PIN label).

    The ligands come from ``rules.centre_ligands``; the name ships only if the strict radical read-back
    (``radicals._ship``: OPSIN's canonical SMILES equal to the input's, charges and hydrogens included)
    accepts it. Only the molecule the outermost ``name`` call was given is named
    (``universal_halane.names_the_input``). Returns '' on any decline."""
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    center = charged[0]
    entry = _LAMBDA_IDE.get(center.GetSymbol())
    if (entry is None or center.GetFormalCharge() != -1 or center.GetNumRadicalElectrons() or center.IsInRing()
            or center.GetIsotope() or any(b.GetBondType() != Chem.BondType.SINGLE for b in center.GetBonds())
            or center.GetDegree() + center.GetTotalNumHs() != entry[1]):
        return ''
    from ..assembly.universal_halane import names_the_input
    from .radicals import _centre_extends_to_chain
    if not names_the_input(mol) or _centre_extends_to_chain(center):
        return ''
    from ..metrics.provenance import record_non_pin_label
    from .centre_ligands import centre_name
    from .ions import _elide_terminal_e
    from .lambda_convention import LAMBDA
    from .radicals import _ship
    ide = f"{LAMBDA}{entry[1] + 1}-{_elide_terminal_e(entry[0])}ide"
    name = centre_name(mol, center.GetIdx(), f"-{ide}", ide)
    # a stereodescriptor in a ligand lowers the name below the PIN at a general tier and declines at the PIN tier
    # (``_ship``, brief item B10); the lambda name is below the PIN in any case
    shipped = _ship(mol, name, False) if name else ''
    if shipped:
        record_non_pin_label(shipped, rule="P-72.3", detail="method (2), an 'ide' on the lambda parent")
    return shipped
