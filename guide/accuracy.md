# Accuracy

Orthonym is judged by what it gets **exactly right** and by what it gets **wrong**.

## The three measures

**Round-trip exact match.** Name the structure, have OPSIN read the name back, and compare the full InChIKey of what OPSIN read with the input's. Every molecule counts. A declined molecule counts as a miss, and so does a name OPSIN cannot read.

**Wrong structures.** An emitted name that OPSIN reads back to a different constitution. The design target is zero. Names that OPSIN cannot read are reported separately, in the column "Unreadable name"; they are never counted as right.

**Preferred-name conformance.** Names compared, character for character apart from spacing and superscript digits, with the worked examples that the IUPAC 2013 recommendations mark as the preferred name (PIN) or the preselected name and that OPSIN can turn into a structure, measured with the `best-effort` tier switched on.

## Measured results

Orthonym 1.0.0 at the `best-effort` tier, every name read back by OPSIN 2.9.0 with its radical option and compared by full InChIKey.

| Set | Molecules | Named (round-trip exact) | Declined | Unreadable name | Other layers |
|:--|--:|--:|--:|--:|--:|
| Held-out ChEBI 2,000 | 2,000 | 1,991 (99.55 %) | 9 | 0 | 0 |
| QM9 | 133,885 | 133,860 (99.98 %) | 25 | 0 | 0 |
| ChEBI | 111,843 | 107,686 (96.28 %) | 4,118 | 39 | 0 |
| PubChem 500,000 | 500,000 | 499,171 (99.83 %) | 829 | 0 | 0 |
| ZINC22 500,000 | 500,000 | 488,407 (97.68 %) | 11,593 | 0 | 0 |
| PubChem 1,000,000 | 1,000,000 | 957,523 (95.75 %) | 42,468 | 1 | 8 |

The columns other than Molecules add up to Molecules. Named (round-trip exact): the name read back to the input's full InChIKey; the percentage is of Molecules. Declined: no name, with a reason. Unreadable name: an emitted name OPSIN could not read; all are names from the engine's exact-match lists (metal tetrapyrrole complexes and a natural-product parent), which OPSIN does not know. Other layers: the same constitution with a different protonation state. Wrong structures, a name that reads back to a different constitution: none on any set.

The comparison with other generators and the full analysis are in the paper.

## Where the numbers come from

The counts are the Orthonym rows of the run records deposited with the paper's data (`run_records/RESCORE_SUMMARY.md`). The held-out ChEBI set is a split of ChEBI structures that no development run had read. The site build recomputes every percentage in the table from the counts in `guide/_data/accuracy.json`, checks that each row adds up, and stops if the table disagrees.
