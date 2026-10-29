"""The ion classes that no producer names, offered last.

``route_charged`` is a cascade of emitters, each for one class of ion, and the producers after it (the universal namer, the
general engine, the rescues of ``Orthonym.name``) name what it declines, each in its own way. The emitters here are for
classes none of those names; each is in a module of its own and reads its name back through OPSIN to the input itself before
it ships.

They are the last rescue of ``Orthonym.name``: asked only for a molecule that every producer and every earlier rescue left
without a name, so no name that ships can change (offer-not-return, a project rule), and each declines at once (a pattern test on
the ion, no naming) for a molecule that is not its class. The router is not the place to ask them: the producers after it
name some of these shapes by another route ('methane(SO-thioperoxolate)' for ``CS[O-]``, '[methyl(oxido)amino]methane' for
``CN(C)[O-]``, a name for ``COP(=O)([S-])OC``), and an emitter asked inside the router would replace those names.
"""
from typing import Optional

from rdkit import Chem


def _ask(emitter, mol, style: str) -> str:
    """The name ``emitter`` gives ``mol``, '' on a decline or a failure (an emitter that fails names nothing; the hang budgets are
    BaseExceptions and pass)."""
    try:
        return emitter(mol, style)
    except Exception:   # noqa: BLE001
        return ''


def late_charged_name(smiles: str, style: str = 'pin') -> Optional[str]:
    """The name of the charged molecule ``smiles`` that the first emitter of this module has for it, or None. The emitters are
    asked in this order, the specific class first: the ester anion of a thio acid ('O,O-diethyl phosphorothioate', not the
    oxidanide of its group), the chalcogen anion of a heteroatom group, the azanide of a carbon group, the centre of several
    charges. The producer id of each is a literal (``tests/unit/metrics/test_producer_trail_static.py``)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or not any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return None
    from ..assembly.universal_halane import names_the_input
    if not names_the_input(mol):
        return None   # a piece a nested frame derived from the input: the outer name is built from its names
    from ..metrics.producer_trail import noted
    from .hetero_group_anions import carbon_group_azanide, hetero_group_chalcogenide
    from .multiply_charged_centres import multiply_charged_centre
    from .thiophosphate_ester_anions import thiophosphate_ester_anion
    name = _ask(thiophosphate_ester_anion, mol, style)
    if name:
        return noted("charged_leftovers:thiophosphate_ester_anion", name, mol)
    name = _ask(hetero_group_chalcogenide, mol, style)
    if name:
        return noted("charged_leftovers:hetero_group_chalcogenide", name, mol)
    name = _ask(carbon_group_azanide, mol, style)
    if name:
        return noted("charged_leftovers:carbon_group_azanide", name, mol)
    name = _ask(multiply_charged_centre, mol, style)
    if name:
        return noted("charged_leftovers:multiply_charged_centre", name, mol)
    return None
