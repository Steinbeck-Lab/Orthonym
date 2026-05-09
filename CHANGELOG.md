# Changelog

All notable changes to Orthonym are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/).

## [v18.0] — 2026-05-09

**Milestone:** Multi-Corpus Benchmark + 17-Phase Architectural Push
**Phases shipped:** 145, 145.1, 145.2, 146, 147, 147.1, 148, 148.1, 148.2, 149, 150, 151, 152, 153, 154, 155, 156, 157, 159 (18 of 19; Phase 158 deferred to v19 per ADR-18.08)
**Verdict:** PASS-WITH-CAVEATS (4 of 5 V18 §4 minimum thresholds not met on data; substrate ships clean — see [`v18_final_report.md`]() §10 + §12)

### Changed

- **Multi-corpus benchmark suite** (ChEBI 5,000 + PubChem 2,000 + OPSIN self-test 500 = 7,500 compounds; replaces v17 ChEBI-500 only) — Phase 145 + ADR-18.01
- **Architectural triple:** competitive candidate scoring (Phase 146 + ADR-18.02), IUPAC locants in P-44.1 cascade across all ring types (Phase 147 + ADR-18.03), fused-heterocycle strict-inequality bypass removal (Phase 148 + ADR-18.04)
- **Parent-correctness weight recalibration** (Phases 148.1, 148.2)
- **Algorithmic FR-2.3 base-component selection** (Phase 149 + ADR-18.05)
- **Retained-name catalog 413 → 914** (+501 entries; Phase 150 OPSIN XML import)
- **VB / spiro / ring-assemblies completion** (Phase 151 audit)
- **Handler-level stereo injection** (Phase 152 + Phase 153 advanced stereo)
- **Skeletal-replacement + multiplicative completion** (Phase 154)
- **Phane nomenclature P-26 + indicated H** (Phase 155)
- **OPSIN grammar pre-validation safety net** (Phase 156)
- **11 IUPAC 2013 errata applied / re-verified** (Phase 157; see )
- **Public-facing accuracy claim:** measured per-corpus dual-metric (absolute + ceiling-relative) + salt-class disclosure + HERITAGE 1990 historical context (per Phase 159 CONTEXT D-05 + D-13)

### Added

-  (Phase 157, 429 LOC; PIN compliance for 11 errata)
-  expanded (Phase 153 + Phase 153-02, 351 LOC; CIP rules 3-5 RDKit limitation per ADR-18.07)
-  (Phase 145, 136 LOC; multi-corpus methodology)
-  (Phase 159; compound classes + confidence bands)
-  (Phase 159; per-failure-category tables)
-  (Phase 150; OPSIN parser scope)
- *.md` (Phase 159; 8 standalone ADR files)
-  (Phase 159 skeleton; ~ 300 LOC; per Phase 147.1 STRATEGIC-DECISIONS Decision 3)
- README.md Accuracy section (Phase 159; per CONTEXT D-05 + D-13)
- `CHANGELOG.md` (this file; Phase 159 deliverable)

### Deferred to v19 (per ADR-18.08 + Phase 147.1 STRATEGIC-DECISIONS Decision 3)

- composer.py decomposition (IM-22 + IM-26)
- P-69 organometallics (IM-24)
- ML fallback gate (IM-25; opt-in via `--allow-ml-fallback`)
- P-25.3 functional replacement nomenclature (IM-27)
- Class-first routing (Phase 158 / ADR-18.08)

### Stats (post-Phase-159 multi-corpus benchmark)

| Metric | Value | Source |
|---|---:|---|
| Overall RT absolute | 2027/7500 (27.03%) | post159/benchmark_multi_corpus_summary.json `overall.inchi_rt_count` / `inchi_rt_fraction` |
| Overall RT ceiling-relative | 31.73% | post159 `overall.inchi_rt_ceiling_relative` |
| Overall graded score | 2.7572/5.0 | post159 `overall.graded_total_mean` |
| Overall OPSIN parse rate | 75.04% | post159 `overall.opsin_parse_rate` |
| ChEBI 5000 RT | 1297/5000 (25.94% absolute, 31.52% ceiling-relative) | post159 `per_corpus.chebi_5000` |
| PubChem 2000 RT | 270/2000 (13.50% absolute, 15.23% ceiling-relative) | post159 `per_corpus.pubchem_2000` |
| OPSIN self-test 500 RT | 460/500 (92.00%) | post159 `per_corpus.opsin_selftest_500` |
| Salt-class RT | 40/314 (12.74%) | post159 `overall.salt_breakdown.salt` |
| Non-salt RT | 1987/7186 (27.65%) | post159 `overall.salt_breakdown.non_salt` |
| Tests passing | 14,068 (44 pre-existing failures) | Phase 157 G1 measurement |
| Canary compounds | 1,304 (preserved across all v18 phases per CONTEXT D-21) | tests/integration/canary/ |
| Source modules | 107 | post-Phase-157 HEAD; `find src/orthonym -name '*.py' \| wc -l` |
| Source LOC | ~120,127 | `find src/orthonym -name '*.py' -exec wc -l {} +` |
| Retained names | ~914 | Phase 150 catalog (Phase 150 OPSIN XML expansion 413 → 914) |

### Historical context

Per HERITAGE-1990-insights.md §12: HERITAGE (Wisniewski, Beilstein Institute, *J. Chem. Inf. Comput. Sci.* 1990, 30, 324-332) achieved **61% expert-status agreement** with human nomenclaturists on stereo-bypassed random samples — the first general-purpose structure→IUPAC-name system. v18 explicitly handles stereo (Phases 152, 153); v19 targets ≥ 65% ceiling-relative RT to exceed the 1990 algorithmic-naming benchmark on the modern multi-corpus suite.

Cite-block: every entry references its owning phase per Phase 159 CONTEXT D-25; full detail in [`v18_final_report.md`]() Appendix D.
