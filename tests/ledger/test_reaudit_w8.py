"""Tests for the Wave-8 Phase-0 ledger re-audit driver (scripts/ledger/reaudit_w8.py)."""
from scripts.ledger.reaudit_w8 import classify, resolve_smiles, norm


def test_ledger_error_when_expected_is_a_comparison_note():
    #: expected is the note "morpholine > pyrimidine"; output has morpholine as parent
    assert classify("morpholine > pyrimidine",
                    "4-[(pyrimidin-5-yl)methyl]morpholine",
                    "4-[(pyrimidin-5-yl)methyl]morpholine", True) == "ledger_error"


def test_anisole_expected_is_wrong_output_is_pin():
    # methoxybenzene IS the PIN; ledger 'expected anisole (PIN)' is itself wrong -> valid_not_pin
    assert classify("anisole (PIN)", "methoxybenzene", "methoxybenzene", True) == "valid_not_pin"


def test_fail_closed_when_gated_unknown():
    assert classify("(pyridin-2-yloxy)acetic acid",
                    "unknown organic compound", "ethanoic acid", None) == "fail_closed"


def test_valid_not_pin_when_rt_clean_but_not_retained():
    assert classify("N6-acetyl-L-lysine",
                    "(2S)-6-acetamido-2-aminohexanoic acid",
                    "(2S)-6-acetamido-2-aminohexanoic acid", True) == "valid_not_pin"


def test_truly_open_when_no_roundtrip():
    assert classify("yohimban", "cyclohexane", "cyclohexane", False) == "truly_open"


def test_needs_example_when_expected_none():
    assert classify("None", "unknown organic compound", "unknown organic compound", None) == "fail_closed"
    assert classify(None, "octane", "octane", True) == "needs_example"


def test_implemented_when_output_matches_expected_real_name():
    # a genuinely healed row: current output == the ledger's expected PIN -> ledger flip (implemented)
    assert classify("pentanehydrazide", "pentanehydrazide", "pentanehydrazide", True) == "implemented"


def test_semicolon_alternatives_are_not_ledger_error():
    # ';' joins ALTERNATIVE example names (multi-example rows), NOT a comparison note.
    # A row that emits one alternative is unverified-vs-PIN, not a healed ledger error.
    assert classify("thiourea; thiosemicarbazide", "thiourea", "thiourea", True) == "valid_not_pin"


def test_multi_alt_exact_match_of_one_alternative_is_not_implemented():
    # got a retained alternative (indane) that is NOT confirmed to be the PIN -> stay open for review
    assert classify("indane; 2,3-dihydro-1H-indene", "indane", "indane", True) == "valid_not_pin"


def test_refusal_output_is_fail_closed_even_with_alternatives():
    # a '(not supported)' refusal is a fail-closed abstention, not an implemented row,
    # even though expected joins alternatives with ';'
    assert classify("sodium (R)-x; sodium (S)-y",
                    "sodium compound (not supported)",
                    "sodium compound (not supported)", None) == "fail_closed"


def test_same_note_routes_to_review_not_flipped():
    # 'same' is not a target name; a real rt-clean output routes to valid_not_pin (review), never flipped
    assert classify("same", "(1r,4r)-1,4-dimethylcyclohexane",
                    "(1r,4r)-1,4-dimethylcyclohexane", True) == "valid_not_pin"


def test_gt_comparison_note_with_wrong_structure_is_not_ledger_error():
    # defensive: a '>' row whose output does NOT round-trip is a real gap, not a ledger error
    assert classify("a > b", "wrongname", "wrongname", False) == "truly_open"


def test_resolve_smiles_unwraps_backticks_and_validates():
    assert resolve_smiles("`OC(=O)C(=O)O`") == "OC(=O)C(=O)O"
    assert resolve_smiles("not a smiles at all") is None


def test_resolve_smiles_picks_first_parseable_of_multi_candidate():
    assert resolve_smiles("`CCCCC(=O)NN`, `CC(=O)NN`") == "CCCCC(=O)NN"


def test_norm_collapses_whitespace_and_case():
    assert norm("  Oxalic   Acid ") == "oxalic acid"
    assert norm(None) == ""


def test_reaudit_end_to_end_small(monkeypatch):
    import json, tempfile, os
    from scripts.ledger.reaudit_w8 import reaudit
    # conftest autouse-disables the OPSIN validity gate for speed; the re-audit driver's
    # whole premise is gated-vs-raw divergence, so re-enable the production gate here.
    from orthonym import namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    rows = [
        {"bb_ref": "P-63.2.3", "capability": "anisole", "status": "OPEN",
         "evidence_smiles": "COc1ccccc1", "expected": "`anisole (PIN)`", "actual": None, "wave": 1,
         "code_locus": None, "verified_at": "old"},
        {"bb_ref": "P-45.6", "capability": "aryloxy leak", "status": "OPEN",
         "evidence_smiles": "OC(=O)COc1ccccn1", "expected": "`[(pyridin-2-yl)oxy]acetic acid`", "actual": None,
         "wave": 2, "code_locus": None, "verified_at": "old"},
        {"bb_ref": "P-101.2.7", "capability": "yohimban", "status": "OPEN",
         "evidence_smiles": "C1CCCCC1", "expected": "`yohimban`", "actual": None, "wave": 7,
         "code_locus": None, "verified_at": "old"},
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    out = reaudit(path, head_sha="testsha", opsin=True)
    by_ref = {r["bb_ref"]: r for r in out["rows"]}
    assert by_ref["P-63.2.3"]["bucket"] in ("ledger_error", "valid_not_pin")  # methoxybenzene is the PIN
    # F-spell-oxy (2026-08-08): the aryloxy leak is HEALED — the row now names at
    # PIN '[(pyridin-2-yl)oxy]acetic acid' (was the atom-drop 'ethanoic acid').
    assert by_ref["P-45.6"]["bucket"] == "implemented"                        # aryloxy now names at PIN
    assert by_ref["P-45.6"]["current_output_raw"] == "[(pyridin-2-yl)oxy]acetic acid"
    assert by_ref["P-45.6"]["owning_phase"] == "P2"
    assert out["summary"]["total_audited"] == 3
    assert set(out["summary"]["by_bucket"]) <= {
        "implemented", "ledger_error", "valid_not_pin", "fail_closed", "truly_open", "needs_example"}


def test_flip_is_line_preserving_and_ascii():
    import json, tempfile, os
    from scripts.ledger.reaudit_w8 import apply_ledger_flips
    lines = [
        json.dumps({"bb_ref": "P-63.2.3", "status": "OPEN", "expected": "anisole", "verified_at": "old"}, ensure_ascii=True),
        json.dumps({"bb_ref": "P-101.2.7", "status": "OPEN", "expected": "yohimban", "verified_at": "old"}, ensure_ascii=True),
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    open(path, "w").write("\n".join(lines) + "\n")
    flips = [{"bb_ref": "P-63.2.3", "bucket": "ledger_error"}]  # only this one flips
    n = apply_ledger_flips(path, flips, head_sha="newsha")
    assert n == 1
    out = [json.loads(l) for l in open(path) if l.strip()]
    assert len(out) == 2                               # no rows added/dropped
    assert out[0]["status"] == "IMPLEMENTED" and out[0]["verified_at"] == "newsha"
    assert out[1]["status"] == "OPEN"                  # untouched


def test_flip_only_flips_implemented_or_ledger_error_buckets():
    import json, tempfile, os
    from scripts.ledger.reaudit_w8 import apply_ledger_flips
    line = json.dumps({"bb_ref": "P-45.6", "status": "OPEN", "expected": "x", "verified_at": "old"}, ensure_ascii=True)
    fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    open(path, "w").write(line + "\n")
    n = apply_ledger_flips(path, [{"bb_ref": "P-45.6", "bucket": "fail_closed"}], head_sha="newsha")
    assert n == 0                                      # fail_closed never flips the ledger
    assert json.loads(open(path).read())["status"] == "OPEN"


def test_flip_disambiguates_duplicate_bb_ref_by_smiles():
    import json, tempfile, os
    from scripts.ledger.reaudit_w8 import apply_ledger_flips
    lines = [
        json.dumps({"bb_ref": "P-65", "status": "OPEN", "evidence_smiles": "CC(=O)O", "verified_at": "old"}, ensure_ascii=True),
        json.dumps({"bb_ref": "P-65", "status": "OPEN", "evidence_smiles": "OC(=O)C(=O)O", "verified_at": "old"}, ensure_ascii=True),
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    open(path, "w").write("\n".join(lines) + "\n")
    n = apply_ledger_flips(path, [{"bb_ref": "P-65", "smiles": "OC(=O)C(=O)O", "bucket": "ledger_error"}], head_sha="s")
    assert n == 1
    out = [json.loads(l) for l in open(path) if l.strip()]
    assert out[0]["status"] == "OPEN"                  # CC(=O)O row untouched
    assert out[1]["status"] == "IMPLEMENTED"           # only the OC(=O)C(=O)O row flipped
