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

`orthonym.__version__` is the installed version, a string such as `'1.0.2'`.
