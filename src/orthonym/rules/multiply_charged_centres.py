"""A mononuclear Group 14 or 15 centre that carries a charge of two or more: R-P(2-), R-Si(3+).

The centre is the parent hydride of the ion and every group on it is a detachable prefix, named by
``rules.centre_ligands`` (the substituent-prefix namer of the substitutive path, with the atom-coverage veto) and
cited as the groups of a mononuclear hydride are, the Blue Book).

* The anion: "Anions derived from parent hydrides and their derivatives" (the Blue Book): "An
  anion derived formally by the removal of one or more hydrons from any position of a neutral parent hydride is
  preferably named by using the suffix '-ide'... Numerical prefixes 'di', 'tri', etc. are used to denote
  multiplicity" (:40902; 'diphenylmethanediide (PIN)',:40910), the 'e' of the hydride kept before the 'd' ('azanediide',
  :40945). R-P(2-) is the phosphane that lost two hydrons: 'phenylphosphanediide'.
* The cation: "Cationic centers in parent hydrides" (the Blue Book; the paragraph that begins "According to the
  general method",:41559, stands under the heading "Specific method",:41553, in this copy): "Di- and polycations
  formally derived by the removal of two or more hydride ions from the parent hydride are named by using the suffix 'ylium' and the
  multiplying prefixes 'bis', 'tris', etc." The book's example of the form, 'ethylazanebis(ylium) (PIN)' (:41672), stands under
   "Cationic centers on other characteristic groups" (:41656: "All other cationic centers are named by substituting the
  appropriate parent cation"), not under. R-Si(3+) is the silane that lost three hydride ions: 'ethynylsilanetris(ylium)'.

The Blue Book gives no example of either on a mononuclear heteroatom hydride with a group on it beyond those
above, so the names are recorded as no PIN. Each ships only when OPSIN reads it back to the input itself.
"""
from rdkit import Chem

#: element -> (parent hydride, standard bonding number,
_ANION_PARENT = {'P': ('phosphane', 3), 'As': ('arsane', 3), 'Sb': ('stibane', 3),
                 'Si': ('silane', 4), 'Ge': ('germane', 4)}
_CATION_PARENT = {'Si': ('silane', 4), 'Ge': ('germane', 4)}
_MULTIPLE_IDE = {2: 'diide', 3: 'triide'}
_MULTIPLE_YLIUM = {2: 'bis(ylium)', 3: 'tris(ylium)'}


def multiply_charged_centre(mol, style: str = 'pin') -> str:
    """The name of ``mol``, one centre with the charge -2/-3 (P, As, Sb, Si, Ge) or +2/+3 (Si, Ge) and its groups; ''
    on any decline."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or abs(charged[0].GetFormalCharge()) not in (2, 3):
        return ''
    centre = charged[0]
    charge = centre.GetFormalCharge()
    entry = (_ANION_PARENT if charge < 0 else _CATION_PARENT).get(centre.GetSymbol())
    if (entry is None or centre.IsInRing() or centre.GetNumRadicalElectrons() or centre.GetIsotope()
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in centre.GetBonds())
            or centre.GetDegree() + centre.GetTotalNumHs() + abs(charge) != entry[1]
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return ''
    from .centre_ligands import centre_name
    suffix = entry[0] + (_MULTIPLE_IDE[-charge] if charge < 0 else _MULTIPLE_YLIUM[charge])
    name = centre_name(mol, centre.GetIdx(), suffix, suffix)
    from .charged_router import _charge_sites_rt_ok
    if not name or not _charge_sites_rt_ok(name, mol):
        return ''
    from ..metrics.provenance import record_non_pin_fragment
    record_non_pin_fragment(name)
    return name
