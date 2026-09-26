# When Orthonym cannot name a molecule

It says so. The plain call returns a label in place of a name, such as
`inorganic compound (not supported)`, and `--provenance` marks the row `abstain` and says why
in `limit_code`:

```console
$ orthonym "O=[U](=O)=O" --provenance | python -m json.tool
{
    "name": "inorganic compound (not supported)",
    "tier": "abstain",
    "is_pin": false,
    "source": "abstain",
    "opsin": "n/a",
    "gates_passed": [],
    "gate_outcome": "suppressed",
    "formula": "O3U",
    "limit_code": "UNSUPPORTED_ELEMENT",
    "stereo_unexpressed": false,
    "suffix_free_prefix_name": false,
    "verified": "unverified"
}
```

In code, `orthonym.errors.is_failure_name(name)` tells such a label from a name:

```python
from orthonym import name_compound
from orthonym.errors import is_failure_name

name = name_compound("O=[U](=O)=O")
is_failure_name(name)   # True
```

A decline is never silent: a molecule that gets no name gets a reason instead. The
[output tiers](../README.md#output-tiers) describe how the names that do pass are labelled.
