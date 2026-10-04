# Python API

The public names of the `orthonym` package. Everything else is [internal](internals/index.md); the counters of `Orthonym` (`get_validation_stats`, `get_dispatch_stats` and the others) are documented there, under `orthonym.namer`.

| Name | What it is for |
|:--|:--|
| [`name_compound`](#orthonym.name_compound) | One molecule, one name |
| [`Orthonym`](#orthonym.Orthonym) | All options, one engine for many molecules |
| [`Orthonym.name_tiered`](#orthonym.Orthonym.name_tiered) | The provenance row: the name, its tier, and every field of how it was checked |
| [`name_with_tree`](#orthonym.name_with_tree) | The name and the tree of its parts |
| [`NamingResult`](#orthonym.NamingResult), [`NameTreeNode`](#orthonym.NameTreeNode) | What `name_with_tree` returns |
| [`classify_limit`](#orthonym.classify_limit), [`OrthonymLimitError`](#orthonym.OrthonymLimitError) | Why a structure is out of scope |
| [`is_failure_name`](#orthonym.errors.is_failure_name) | Tell a label from a name |

Every example below is the engine's own output.

(python-input)=
## Input: a SMILES string or an RDKit molecule

Every function and method here that names a structure takes a SMILES string or an RDKit `Chem.Mol`: `name_compound`, `name_with_tree`, `classify_limit`, and the `Orthonym` methods `name`, `name_tiered`, `name_with_tree` and `name_with_confidence`. The parameter is called `smiles` for both.

```pycon
>>> from rdkit import Chem
>>> from orthonym import name_compound
>>> name_compound(Chem.MolFromSmiles("C[C@H](O)CC"))
'(2S)-butan-2-ol'
```

A Mol is named through RDKit's SMILES of it. Orthonym writes `Chem.MolToSmiles(mol)` from a copy of your Mol (your Mol is not changed), checks that this SMILES holds the same molecule and the same stereo as the Mol, and then names the SMILES exactly as it names a string you pass. So a Mol gets the result that `Chem.MolToSmiles(mol)` gets: the same name, the same provenance row, the same label when there is no name.

The check reads the Mol as RDKit reads it, and the call raises `ValueError` when RDKit's SMILES of the Mol holds something else:

- RDKit must read the SMILES back.
- RDKit's standard InChI of the Mol must be the InChI of the SMILES. The Mol's InChI is made from its stereo tags, without its coordinates, and before anything in the Mol is cleaned up.
- Atom for atom, the SMILES must have the Mol's atoms, bonds, hydrogen counts and charges, and every stereocentre and double bond whose configuration the Mol's tags set must have that configuration in the SMILES, and the SMILES must set no other. This also covers stereo that InChI does not see, such as the C=N of an amidine.
- A lone-pair stereocentre (a sulfoxide, a phosphine, an aziridine nitrogen) must read the same in RDKit's reading of the SMILES as in the standard reading of SMILES, which is how Orthonym names a SMILES string. RDKit reads such a centre otherwise when it is written first or carries a ring-closure digit at some positions.
- A Mol with an AND (racemic) or OR (unknown enantiomer) stereo group, from CXSMILES or a V3000 record, is declined, because a SMILES holds no stereo group and the molecule would be named as one enantiomer. A Mol with an atropisomer tag (`STEREOATROPCW`, `STEREOATROPCCW`) is declined, because no SMILES holds the configuration of an axis, and so is a Mol with a `CHI_OTHER` chiral tag. Square-planar, trigonal-bipyramidal and octahedral tags pass: a SMILES holds them, and no name states them, so such a Mol (cisplatin, for one) gets the result of its SMILES.

The message of the `ValueError` says which part failed. Mols that RDKit's readers make (`MolFromSmiles`, `MolFromMolBlock`, `SDMolSupplier`, `MolFromInchi`, with or without hydrogen atoms) almost always pass. Mols built or edited in code fail more often, because RDKit's own InChI and SMILES writers can read their tags in different ways. Two examples: a double bond whose configuration is only in the `/` and `\` directions of `Chem.MolFromSmiles(smiles, sanitize=False)` (RDKit's InChI of that Mol has no configuration, its SMILES has one), and a double bond left open between two double bonds whose configuration is set (any SMILES of the molecule can give the open bond a configuration). Fix the tags, or name a SMILES you have checked yourself.

Stereo comes from the Mol's stereo tags, not from its coordinates. RDKit sets the tags when it reads a mol block or an SDF record; for a Mol you build from 3D coordinates, call `Chem.AssignStereochemistryFrom3D(mol)` first. A Mol made by `Chem.MolFromSmiles` holds RDKit's reading of the string. At a lone-pair stereocentre written first, as in `[S@@](C)(=O)c1ccccc1`, RDKit reads the other configuration than the standard reading, so that Mol and that string get the names of two enantiomers.

To name the molecules of an SDF file:

```python
from rdkit import Chem
from orthonym import name_compound

for mol in Chem.SDMolSupplier("compounds.sdf"):
    if mol is None:  # RDKit could not read this record
        continue
    print(mol.GetProp("_Name"), name_compound(mol))
```

Because the SMILES is what gets named, the fields of a result that repeat the input (the `smiles` of an `OrthonymLimitError`, and of the `limit` record of `name_with_confidence`) hold `Chem.MolToSmiles(mol)`, and atom indices in a result (`atom_to_locant_hint`, `atom_to_locant`) refer to the atoms of `Chem.MolFromSmiles(Chem.MolToSmiles(mol))`, not to your Mol's own atom order. For a Mol without hydrogen atoms, `Chem.MolToSmiles(mol)` leaves the property `_smilesAtomOutputOrder` on `mol`, which lists your Mol's atom for each of those indices.

`None`, a Mol that RDKit cannot sanitize, and a Mol the check declines raise `ValueError`, as a SMILES that RDKit cannot read does; `classify_limit` returns `None` for them. Any other type raises `TypeError`. An empty Mol is named as the empty string is. These rules apply to the calls your code makes. Orthonym's own calls to these functions, made while it names a molecule, keep their earlier handling.

## Naming

```{eval-rst}
.. autofunction:: orthonym.name_compound

.. autoclass:: orthonym.Orthonym
   :members: name, name_with_tree, name_with_confidence
   :exclude-members: name_tiered, get_validation_stats, get_dispatch_stats, get_inner_dispatch_stats, reset_dispatch_stats
```

## The provenance fields

```{eval-rst}
.. automethod:: orthonym.Orthonym.name_tiered
```

## The parts of a name

```{eval-rst}
.. autofunction:: orthonym.name_with_tree

.. autoclass:: orthonym.NamingResult
   :no-members:

.. autoclass:: orthonym.NameTreeNode
   :no-members:
```

## Declines

```{eval-rst}
.. autofunction:: orthonym.classify_limit

.. autoexception:: orthonym.OrthonymLimitError
   :no-members:

.. autofunction:: orthonym.errors.is_failure_name
```

## Version

`orthonym.__version__` is the installed version, a string of the form `'X.Y.Z'`.
