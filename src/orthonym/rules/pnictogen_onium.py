"""A substituted arsanium, stibanium or bismuthanium cation (the all-organic program, lane S).

 (the Blue Book): a cation formed by adding a hydron to a mononuclear parent hydride is named with
the suffix 'ium' on the parent hydride name, and its groups are prefixes: '[6-(dimethylsulfaniumyl)hexyl]
tri(methyl)phosphanium (PIN)' (:42358). The engine names a centre whose groups are all hydrocarbyl
('tetramethylarsanium') through ``rules.ions``; a group with a heteroatom ('(2-hydroxyethyl)', '(2-oxobutyl)')
reached no producer, and the substituent namer cannot name the cation as a prefix of another parent, because the
cation is the senior class, "6 Cations"). Phosphorus keeps its own routes.

The groups are named by the trusted ligand namer (``rules.centre_ligands.centre_name``); the caller ships the name
only through its full-InChIKey read-back. Built by this program, the name is recorded as no PIN
(``record_non_pin_fragment``): systematic_verified at most.
"""
from rdkit import Chem

_ONIUM = {'As': 'arsanium', 'Sb': 'stibanium', 'Bi': 'bismuthanium'}


def emit_pnictogen_onium(mol, center_idx: int) -> str:
    """The name of the cation ``mol`` on its As, Sb or Bi centre ``center_idx`` (charge +1, four single bonds with
    its hydrogens, in no ring, at least one group holding an atom other than carbon), or ''."""
    centre = mol.GetAtomWithIdx(center_idx)
    suffix = _ONIUM.get(centre.GetSymbol())
    if (suffix is None or centre.GetFormalCharge() != 1 or centre.IsInRing() or centre.GetNumRadicalElectrons()
            or centre.GetDegree() + centre.GetTotalNumHs() != 4
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in centre.GetBonds())):
        return ''
    ligand_halogens = {nb.GetIdx() for nb in centre.GetNeighbors() if nb.GetSymbol() in ('F', 'Cl', 'Br', 'I')}
    others = [a for a in mol.GetAtoms() if a.GetIdx() != center_idx and a.GetAtomicNum() > 1
              and a.GetIdx() not in ligand_halogens]
    if all(a.GetSymbol() == 'C' for a in others):
        return ''       # hydrocarbyl and halogen groups: ``rules.ions`` names them ('chlorotri(methyl)arsanium')
    from .centre_ligands import centre_name
    name = centre_name(mol, center_idx, suffix, suffix)
    if name:
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)
    return name
