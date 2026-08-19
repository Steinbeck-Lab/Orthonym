from orthonym.assembly.substituent_naming import _build_alkenyl_name


def test_octaene_substituent_uses_word_not_digit():
    # 32-carbon chain, attach at 1, eight C=C (a polyprenyl/solanesyl-type arm)
    name = _build_alkenyl_name(32, 1, [2, 6, 10, 14, 18, 22, 26, 30], [])
    assert "octaen" in name, name
    assert "8en" not in name, name           # the digit-fallback bug
    assert name.endswith("-1-yl"), name

def test_hexaene_and_nonaene_and_undecaene():
    assert "hexaen" in _build_alkenyl_name(24, 1, [2, 6, 10, 14, 18, 22], [])
    assert "nonaen" in _build_alkenyl_name(36, 1, [2, 6, 10, 14, 18, 22, 26, 30, 34], [])
    assert "undecaen" in _build_alkenyl_name(44, 1, [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42], [])

def test_low_counts_unchanged():
    # di/tri/tetra/penta must still be correct (no regression)
    assert "dien" in _build_alkenyl_name(6, 1, [2, 4], [])
    assert "trien" in _build_alkenyl_name(8, 1, [2, 4, 6], [])
    assert "tetraen" in _build_alkenyl_name(10, 1, [2, 4, 6, 8], [])

def test_polyyne_multiplier_word_not_digit():
    # triple-bond path at :571 has the identical bug
    name = _build_alkenyl_name(20, 1, [], [2, 6, 10, 14, 18, 22])
    assert "hexayn" in name, name
    assert "6yn" not in name, name
