# Accuracy

Orthonym is judged by what it gets **exactly right** and by what it gets **wrong**.

## The three measures

**Round-trip exact match.** Name the structure, have OPSIN read the name back, and compare the full InChIKey of what OPSIN read with the input's. Every molecule counts. A declined molecule counts as a miss, and so does a name OPSIN cannot read.

**Wrong structures.** An emitted name that OPSIN reads back to a different constitution. The design target is zero. Names that OPSIN cannot read are reported separately, in the column "Unreadable name"; they are never counted as right.

**Preferred-name conformance.** Names compared, character for character apart from spacing and superscript digits, with the worked examples that the IUPAC 2013 recommendations mark as the preferred name (PIN) or the preselected name and that OPSIN can turn into a structure, measured with the `best-effort` tier switched on.

## The two tiers

Orthonym answers at two tiers, and they name different numbers of molecules.

The **default** is the `best-effort` tier. It returns the verified Preferred IUPAC Name (PIN) when the strict PIN path builds it, and otherwise a systematic name that is not the PIN, checked by the same round trip and labelled with its tier. The **PIN-only tier** (`--emit-tier pin`, the default of releases up to 1.0.x) returns a verified PIN, or no name and the reason. It declines a molecule when it cannot verify a PIN, even when a correct name that is not the PIN exists. Ask for it with `--emit-tier pin`, with `--batch FILE --emit-tier pin` for a file of SMILES, or in Python with `name_compound(smiles, emit_tier="pin")` ([Output tiers](tiers.md)). The PIN-only tier is a rule about which names to return. It is not a limit of what the engine can name.

The table gives both for the same molecules. It is Orthonym 1.0.9 with the changes merged after it. OPSIN 2.9.0 read each name back, and a name counts as named only when it read back to the input's molecule.

| Set | Molecules | Named, default | Named, PIN-only tier | Declined, default | Declined, PIN-only tier | Unreadable, default | Unreadable, PIN-only tier |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Held-out ChEBI 2,000 | 2,000 | 1,994 (99.70 %) | 1,062 (53.10 %) | 5 | 935 | 1 | 3 |
| ChEBI development 2,000 | 2,000 | 1,993 (99.65 %) | 1,043 (52.15 %) | 4 | 953 | 3 | 4 |
| Drug-like PubChem 2,000 | 2,000 | 1,959 (97.95 %) | 866 (43.30 %) | 41 | 1,121 (+1 timeout) | 0 | 12 |

Each tier's columns add up to Molecules. Named: the name read back to the input's full InChIKey; the percentage is of Molecules. Declined: no name, with a reason. A timeout (naming one molecule did not finish within the 60 seconds the harness allows) is also no name and counts as a miss. Unreadable: the engine emitted a name that OPSIN 2.9.0 cannot read, so it does not count as named. These are the documented exceptions of [Output tiers](tiers.md): at the PIN-only tier, for example, a PIN spelling that OPSIN's grammar lacks or a PIN with stereodescriptors OPSIN cannot read; at the default, a name with stereodescriptors OPSIN cannot read for a molecule that has no other name. Wrong structures, a name that reads back to a different molecule: none in any cell.

The two ChEBI sets are samples of ChEBI structures that have at most 50 heavy atoms, one fragment, carbon, no radicals, no isotopes and no element other than C, H, N, O, P, S, F, Cl, Br and I. The held-out set is a split that no development run had read. The development set is a sample that development runs did read, so it is not an independent test. The drug-like set is 2,000 drug-like PubChem structures.

On the held-out set the default names 1,994 of 2,000 molecules (99.70 %) and declines 5, and the PIN-only tier names 1,062 (53.10 %) and declines 935. The PIN-only tier declines the names built on a semisystematic name of the recommendations' Chapter P-10 (`D-glucose`, `glycylglutamylalanine`), for which no PIN is identified (P-100); the default returns them.

## Measured results

Orthonym 1.0.0 at the `best-effort` tier, every name read back by OPSIN 2.9.0 with its radical option and compared by full InChIKey. This is the paper's table. The section above has both tiers for a newer version.

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

The counts are the Orthonym rows of the run records deposited with the paper's data (`run_records/RESCORE_SUMMARY.md`). The held-out ChEBI set is a split of ChEBI structures that no development run had read. The counts of the two-tier table come from a run of the project's evaluation harness at both tiers, on the sets named in the table. The site build recomputes every percentage in both tables from the counts in `guide/_data/accuracy.json`, checks that each row adds up, and stops if a table disagrees.
