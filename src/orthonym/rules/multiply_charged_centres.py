"""A mononuclear Group 14 or 15 centre that carries a charge of two or more: R-P(2-), R-Si(3+), R2Sn(2+).

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
#: the Group 14 metals named on their parent hydrides; 'stannane, SnH4, and plumbane, PbH4',,
#: the Blue Book), each a cation centre only where ``data/metal_scope.py`` admits it to PARENT_HYDRIDE_IONS
_METAL_CATION_PARENT = {'Sn': ('stannane', 4), 'Pb': ('plumbane', 4)}


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
    if entry is None and charge > 0 and centre.GetSymbol() in _METAL_CATION_PARENT:
        from ..data.metal_scope import PARENT_HYDRIDE_IONS, has_mechanism
        if has_mechanism(centre.GetSymbol(), PARENT_HYDRIDE_IONS):
            entry = _METAL_CATION_PARENT[centre.GetSymbol()]
    if (entry is None or centre.IsInRing() or centre.GetNumRadicalElectrons() or centre.GetIsotope()
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in centre.GetBonds())
            or centre.GetDegree() + centre.GetTotalNumHs() + abs(charge) != entry[1]
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return ''
    from .centre_ligands import centre_name
    from .charged_router import _charge_sites_rt_ok
    suffix = entry[0] + (_MULTIPLE_IDE[-charge] if charge < 0 else _MULTIPLE_YLIUM[charge])
    suffixes = [suffix]
    if charge == 2 and centre.GetSymbol() in _METAL_CATION_PARENT:
        # OPSIN reads a substituted tin or lead centre of charge 2+ as divalent and reads 'dimethylstannanebis(ylium)'
        # as nothing; with the standard bonding number cited ('dimethyl-λ4-stannanebis(ylium)') it reads the input. λ4
        # is the standard bonding number of tin and lead "Standard bonding numbers", Table 1.3), so the
        # citation is redundant, not wrong; it is offered only after the book form, which a reader that takes it
        # ships first (unsubstituted 'stannanebis(ylium)' keeps the bare form)
        from .lambda_convention import LAMBDA
        suffixes.append(f"-{LAMBDA}4-{suffix}")
    for suf in suffixes:
        name = centre_name(mol, centre.GetIdx(), suf, suffix)
        if name and _charge_sites_rt_ok(name, mol):
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(name)
            return name
    return ''
